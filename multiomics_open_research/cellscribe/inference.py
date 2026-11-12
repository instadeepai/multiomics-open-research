# Copyright 2026 InstaDeep Ltd
#
# Licensed under the Creative Commons BY-NC-SA 4.0 License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""High-level API: raw counts in, cell types / cell states out."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import torch

from multiomics_open_research.cellscribe.modeling_cellscribe import (
    CellScribe,
    resolve_path,
)
from multiomics_open_research.cellscribe.postprocessing import (
    extract_predicted_cell_type,
    parse_cell_state_prediction,
)
from multiomics_open_research.cellscribe.preprocessing import (
    CELL_TYPES_NAME,
    GENE_LIST_NAME,
    GeneMapper,
    ScFoundationPreprocessor,
    load_gene_list,
)
from multiomics_open_research.cellscribe.prompts import (
    TASK_CELL_STATE,
    TASK_CELL_TYPING,
    build_in_context_prompt,
    build_prompt,
)

# Mixed precision of the predictor: "auto", None (fp32) or a torch dtype.
AutocastDtype = Union[str, torch.dtype, None]


@dataclass
class CellStatePrediction:
    programs: list[tuple[str, float]]  # (meta-program, score), strongest first
    hallmark: Optional[str]
    text: str


class CellScribePredictor:
    """
    >>> predictor = CellScribePredictor.from_pretrained("path/or/hub-id", device="cuda")
    >>> predictor.predict_cell_type(counts, gene_names)
    """

    def __init__(
        self,
        model: CellScribe,
        tokenizer,
        gene_list: Sequence[str],
        cell_types: Optional[Sequence[str]] = None,
    ) -> None:
        """``cell_types``: label vocabulary used to read cell-type answers (the
        training vocabulary, as in the paper's evaluation). ``None`` returns the
        free text after "The cell type is"."""
        self.model = model
        self.cell_types = list(cell_types) if cell_types else None
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.mapper = GeneMapper(gene_list)
        cfg = model.config.scfoundation
        self.preprocessor = ScFoundationPreprocessor(
            num_genes=len(gene_list) + 2,
            pad_token_id=cfg["pad_token_id"],
            target_res_str=cfg["tgthighres"],
        ).to(self.device)

    @classmethod
    def from_pretrained(
        cls, path: Union[str, Path], device: Union[str, torch.device] = "cpu"
    ) -> "CellScribePredictor":
        from transformers import AutoTokenizer

        path = resolve_path(path)
        model = CellScribe.from_pretrained(path, device=device)
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        cell_types = (
            load_gene_list(path / CELL_TYPES_NAME)
            if (path / CELL_TYPES_NAME).is_file()
            else None
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            gene_list=load_gene_list(path / GENE_LIST_NAME),
            cell_types=cell_types,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _autocast(self, autocast_dtype: AutocastDtype) -> torch.autocast:
        """Mixed-precision context. ``"auto"``: bfloat16 on GPU (the precision of
        the paper's evaluation, and half the encoder memory), fp32 on CPU."""
        if autocast_dtype == "auto":
            autocast_dtype = torch.bfloat16 if self.device.type == "cuda" else None
        if isinstance(autocast_dtype, str):
            raise ValueError(
                "autocast_dtype must be 'auto', None or a dtype, "
                f"got {autocast_dtype!r}"
            )
        if autocast_dtype is not None and self.device.type != "cuda":
            # On CPU, torch's TransformerEncoderLayer keeps its fused fast path
            # under autocast, which mixes reduced-precision inputs with fp32 weights.
            raise ValueError("mixed precision is only supported on GPU; use None")
        return torch.autocast(
            self.device.type,
            dtype=autocast_dtype or torch.float32,
            enabled=autocast_dtype is not None,
        )

    @torch.no_grad()
    def generate_text(
        self,
        counts,
        gene_names: Sequence[str],
        task: str = TASK_CELL_TYPING,
        batch_size: int = 32,
        autocast_dtype: AutocastDtype = "auto",
    ) -> list[str]:
        """Generated answer for each cell.

        Args:
            counts: ``(n_cells, n_genes)`` raw UMI counts (numpy or scipy sparse).
            gene_names: HGNC symbols of the columns.
            autocast_dtype: mixed precision. ``"auto"`` (default): bfloat16 on GPU,
                as in the paper's evaluation, and fp32 on CPU; ``None``: fp32;
                or a dtype such as ``torch.float16``.

        The scFoundation pooling runs over the padded batch, so a cell's embedding
        depends slightly on its batch-mates; results are reproducible for a
        fixed ``batch_size`` and cell order.
        """
        if len(gene_names) != counts.shape[1]:
            raise ValueError(f"{len(gene_names)} names for {counts.shape[1]} columns")
        # Genes are matched once; cells are densified one batch at a time, so a
        # large sparse or backed matrix never has to fit in memory as a whole.
        indices = self.mapper.column_indices(gene_names)
        prompt = self.tokenizer(
            [build_prompt(self.tokenizer, task)], return_tensors="pt"
        )
        texts: list[str] = []
        for start in range(0, counts.shape[0], batch_size):
            dense = self.mapper.map_columns(counts[start : start + batch_size], indices)
            cells = self.preprocessor(torch.from_numpy(dense).to(self.device))
            n = cells["x"].shape[0]
            with self._autocast(autocast_dtype):
                generated = self.model.generate(
                    input_ids=prompt.input_ids.expand(n, -1).to(self.device),
                    attention_mask=prompt.attention_mask.expand(n, -1).to(self.device),
                    scrnaseq_preprocessed_input=cells,
                )
            texts += self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return texts

    def predict_cell_type(
        self, counts, gene_names: Sequence[str], **kwargs
    ) -> list[str]:
        texts = self.generate_text(counts, gene_names, task=TASK_CELL_TYPING, **kwargs)
        return [extract_predicted_cell_type(t, self.cell_types) for t in texts]

    def predict_cell_state(
        self, counts, gene_names: Sequence[str], **kwargs
    ) -> list[CellStatePrediction]:
        texts = self.generate_text(counts, gene_names, task=TASK_CELL_STATE, **kwargs)
        predictions = []
        for text in texts:
            programs, hallmark = parse_cell_state_prediction(text)
            predictions.append(CellStatePrediction(programs, hallmark, text))
        return predictions

    @torch.no_grad()
    def embed_cells(
        self,
        counts,
        gene_names: Sequence[str],
        batch_size: int = 32,
        autocast_dtype: AutocastDtype = "auto",
    ) -> torch.Tensor:
        """Conditioning vector of each cell, ``[n_cells, hidden_size]``."""
        if len(gene_names) != counts.shape[1]:
            raise ValueError(f"{len(gene_names)} names for {counts.shape[1]} columns")
        indices = self.mapper.column_indices(gene_names)
        vectors = []
        for start in range(0, counts.shape[0], batch_size):
            dense = self.mapper.map_columns(counts[start : start + batch_size], indices)
            cells = self.preprocessor(torch.from_numpy(dense).to(self.device))
            with self._autocast(autocast_dtype):
                vectors.append(self.model.embed_cells(cells).float())
        return torch.cat(vectors)

    @torch.no_grad()
    def generate_text_in_context(
        self,
        counts,
        gene_names: Sequence[str],
        example_counts,
        example_gene_names: Sequence[str],
        example_labels: Sequence[str],
        batch_size: int = 32,
        autocast_dtype: AutocastDtype = "auto",
    ) -> list[str]:
        """Generated cell-typing answer for each cell, given labelled example cells.

        In-context learning: the prompt lists the example cells with their cell
        types ("Example cell k: The cell type is <label>."), each example line
        being conditioned by its own cell, then asks for the type of the query
        cell. Example labels may be cell types absent from the training data.

        Args:
            counts, gene_names: query cells (as in :meth:`generate_text`).
            example_counts, example_gene_names: example cells, raw counts.
            example_labels: cell type of each example cell, in order.
        """
        if len(example_labels) != example_counts.shape[0]:
            raise ValueError(
                f"{len(example_labels)} labels for {example_counts.shape[0]} examples"
            )
        examples = self.embed_cells(
            example_counts,
            example_gene_names,
            batch_size=batch_size,
            autocast_dtype=autocast_dtype,
        )
        prompt_ids, spans = build_in_context_prompt(self.tokenizer, example_labels)
        input_ids = torch.tensor([prompt_ids], device=self.device)
        texts: list[str] = []
        for start in range(0, counts.shape[0], batch_size):
            queries = self.embed_cells(
                counts[start : start + batch_size],
                gene_names,
                batch_size=batch_size,
                autocast_dtype=autocast_dtype,
            )
            n = queries.shape[0]
            # One conditioning vector per prompt token: the query cell everywhere,
            # except on each example line, conditioned by its example cell.
            plan = queries[:, None, :].repeat(1, len(prompt_ids), 1)
            for k, (span_start, span_end) in enumerate(spans):
                plan[:, span_start:span_end] = examples[k]
            with self._autocast(autocast_dtype):
                generated = self.model.generate(
                    input_ids=input_ids.expand(n, -1),
                    attention_mask=torch.ones_like(input_ids).expand(n, -1),
                    conditioning=plan,
                    conditioning_default=queries,
                )
            texts += self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return texts

    def predict_cell_type_in_context(
        self,
        counts,
        gene_names: Sequence[str],
        example_counts,
        example_gene_names: Sequence[str],
        example_labels: Sequence[str],
        **kwargs,
    ) -> list[str]:
        """Cell type of each query cell, given labelled example cells (see
        :meth:`generate_text_in_context`). The answer is read as free text (the
        text after "The cell type is"), so it can be a type outside
        ``cell_types``."""
        texts = self.generate_text_in_context(
            counts=counts,
            gene_names=gene_names,
            example_counts=example_counts,
            example_gene_names=example_gene_names,
            example_labels=example_labels,
            **kwargs,
        )
        return [extract_predicted_cell_type(t) for t in texts]

    def predict_anndata(
        self, adata, task: str = TASK_CELL_TYPING, layer: Optional[str] = None, **kwargs
    ):
        """Same, for an AnnData with raw counts in ``layer`` (default ``.X``) and
        HGNC symbols as ``var_names``. Cells are read and processed one batch at a
        time, so a large AnnData can be opened with ``backed="r"``."""
        counts = adata.layers[layer] if layer else adata.X
        if task == TASK_CELL_TYPING:
            return self.predict_cell_type(counts, list(adata.var_names), **kwargs)
        return self.predict_cell_state(counts, list(adata.var_names), **kwargs)
