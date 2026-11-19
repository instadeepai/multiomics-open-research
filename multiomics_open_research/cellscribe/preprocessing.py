# Copyright 2026 InstaDeep Ltd and CentraleSupélec
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Raw counts -> scFoundation inputs (same steps as in training)."""

from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch

GENE_LIST_NAME = "gene_list.txt"
CELL_TYPES_NAME = "cell_types.txt"  # training label vocabulary


def load_gene_list(path: Union[str, Path]) -> list[str]:
    """One entry per line (gene symbols in model order, or cell-type labels)."""
    return [g for g in Path(path).read_text().splitlines() if g]


class GeneMapper:
    """Reorders a ``(cells x genes)`` count matrix onto the model's gene list.

    Genes outside the list are dropped, listed genes absent from the input stay
    at zero. ``counts`` may be a numpy array or a scipy sparse matrix.
    """

    def __init__(self, gene_list: Sequence[str]) -> None:
        self.gene_list = list(gene_list)
        self.gene_to_id = {g: i for i, g in enumerate(self.gene_list)}

    def __call__(self, counts, gene_names: Sequence[str]) -> np.ndarray:
        """Dense ``(cells x model genes)`` float32 matrix for ``counts``."""
        if len(gene_names) != counts.shape[1]:
            raise ValueError(f"{len(gene_names)} names for {counts.shape[1]} columns")
        return self.map_columns(counts, self.column_indices(gene_names))

    def column_indices(self, gene_names: Sequence[str]) -> tuple[list[int], list[int]]:
        """``(input columns, model columns)`` of the genes in the model's list."""
        source, target = [], []
        for i, gene in enumerate(gene_names):
            j = self.gene_to_id.get(str(gene))
            if j is not None:
                source.append(i)
                target.append(j)
        if not source:
            raise ValueError("no input gene matches the model gene list (HGNC symbols)")
        return source, target

    def map_columns(self, counts, indices: tuple[list[int], list[int]]) -> np.ndarray:
        """Apply :meth:`column_indices` to a block of cells."""
        source, target = indices
        sub = counts[:, source]
        sub = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
        dense = np.zeros((counts.shape[0], len(self.gene_list)), dtype=np.float32)
        dense[:, target] = sub
        return dense


class ScFoundationPreprocessor(torch.nn.Module):
    """CP10k + log1p, append the two depth tokens, gather the non-zero entries
    and right-pad them to the longest cell of the batch."""

    def __init__(
        self, num_genes: int, pad_token_id: int = 103, target_res_str: str = "t4"
    ) -> None:
        super().__init__()
        self.num_genes = num_genes  # model genes + 2 depth tokens
        self.pad_token_id = pad_token_id
        self.target_res = float(target_res_str[1:])
        self.register_buffer("all_gene_ids", torch.arange(self.num_genes))

    def _gather_and_pad(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = data.shape[0]
        max_len = int(mask.sum(dim=1).max().item())
        padded_data = torch.full(
            (batch_size, max_len),
            float(self.pad_token_id),
            dtype=data.dtype,
            device=data.device,
        )
        for i in range(batch_size):
            expressed_data = data[i, mask[i]]
            padded_data[i, : len(expressed_data)] = expressed_data
        return padded_data

    def forward(self, gene_expression: torch.Tensor) -> dict[str, torch.Tensor]:
        total_counts = gene_expression.sum(dim=1, keepdim=True) + 1e-9
        log_normalized = torch.log1p(gene_expression / total_counts * 10000)

        log_scaled_counts = torch.log10(total_counts.squeeze(1) * self.target_res)
        log_total_counts = torch.log10(total_counts.squeeze(1))
        special_tokens = torch.stack([log_scaled_counts, log_total_counts], dim=1)
        pretrain_gene_x = torch.cat([log_normalized, special_tokens], dim=1)

        value_mask = pretrain_gene_x > 0
        if not bool(value_mask.any(dim=1).all()):
            raise ValueError("every cell needs at least one non-zero count")
        x = self._gather_and_pad(pretrain_gene_x, value_mask)
        batch_gene_ids = self.all_gene_ids.expand(pretrain_gene_x.shape[0], -1)
        return {
            "x": x,
            "x_padding_mask": x == self.pad_token_id,
            "position_gene_ids": self._gather_and_pad(batch_gene_ids, value_mask),
        }
