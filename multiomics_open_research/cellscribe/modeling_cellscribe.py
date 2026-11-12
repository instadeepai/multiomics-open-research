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
"""CellScribe: scFoundation cell encoder + Qwen2.5 conditioned through AdaLN-Zero.

counts -> scFoundation -> pooled embedding [B, 3072]
       -> Linear + LayerNorm -> conditioning vector [B, 896]
prompt -> Qwen2.5-0.5B-Instruct whose layer norms are modulated by the
          conditioning vector -> "The cell type is ..."
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import transformers

from multiomics_open_research.cellscribe.ada_ln_zero import (
    attach_ada_ln_zero,
    create_conditional_model_class,
)
from multiomics_open_research.cellscribe.scfoundation import SCFoundationWrapper

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"


@dataclass
class CellScribeConfig:
    scfoundation: dict  # scFoundation encoder hyper-parameters
    llm: dict  # transformers config of the language model (Qwen2)
    scrnaseq_embedding_dim: int = 3072
    max_new_tokens: int = 80
    model_type: str = "cellscribe"

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CellScribeConfig":
        return cls(**json.loads(Path(path).read_text()))


class LinearHead(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CellScribe(nn.Module):
    def __init__(self, config: CellScribeConfig) -> None:
        super().__init__()
        self.config = config
        self.scrna_encoder = SCFoundationWrapper(config.scfoundation)

        llm_config = transformers.AutoConfig.for_model(**config.llm)
        text_embedding_dim = llm_config.hidden_size
        self.scrna_projector = LinearHead(
            config.scrnaseq_embedding_dim, text_embedding_dim
        )
        self.conditioning_norm = nn.LayerNorm(
            text_embedding_dim, elementwise_affine=False, eps=1e-6
        )

        base_llm_class = getattr(transformers, llm_config.architectures[0])
        conditional_llm = create_conditional_model_class(base_llm_class)(llm_config)
        self.llm = attach_ada_ln_zero(conditional_llm, text_embedding_dim)

    def _transform_conditioning(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.conditioning_norm(self.scrna_projector(embeddings))

    def embed_cells(self, scrnaseq_preprocessed_input: dict) -> torch.Tensor:
        """Preprocessed cells -> conditioning vectors ``[B, hidden_size]``."""
        return self._transform_conditioning(
            self.scrna_encoder(**scrnaseq_preprocessed_input)
        )

    def _get_conditioning(
        self,
        scrnaseq_preprocessed_input: Optional[dict],
        conditioning: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if conditioning is not None:
            return conditioning
        if scrnaseq_preprocessed_input is None:
            raise ValueError(
                "Pass either 'scrnaseq_preprocessed_input' or 'conditioning'."
            )
        return self.embed_cells(scrnaseq_preprocessed_input)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        scrnaseq_preprocessed_input: Optional[dict] = None,
        conditioning: Optional[torch.Tensor] = None,
        conditioning_default: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Teacher-forced pass.

        Give either the preprocessed cells or a ``conditioning`` computed by
        :meth:`embed_cells`: ``[B, H]`` (one cell per sequence) or ``[B, T, H]``
        (one cell per token, for in-context learning; positions past ``T`` use
        ``conditioning_default``, ``[B, H]``).
        """
        conditioning = self._get_conditioning(scrnaseq_preprocessed_input, conditioning)
        return self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            conditioning_signal=conditioning,
            conditioning_default=conditioning_default,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        scrnaseq_preprocessed_input: Optional[dict] = None,
        conditioning: Optional[torch.Tensor] = None,
        conditioning_default: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Greedy generation; returns only the generated tokens.

        ``conditioning`` / ``conditioning_default``: as in :meth:`forward`. With a
        per-token ``[B, T, H]`` plan covering the prompt, the generated tokens are
        conditioned by ``conditioning_default`` (the query cell).
        """
        conditioning = self._get_conditioning(scrnaseq_preprocessed_input, conditioning)
        kwargs.setdefault("max_new_tokens", self.config.max_new_tokens)
        output = self.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            conditioning_signal=conditioning,
            conditioning_default=conditioning_default,
            **kwargs,
        )
        return output[:, input_ids.shape[1] :]

    # ------------------------------------------------------------------ #

    def save_pretrained(self, path: Union[str, Path]) -> None:
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.config.save(path / CONFIG_NAME)
        state = {
            k: v.detach().cpu().contiguous()
            for k, v in self.state_dict().items()
            if k != "llm.lm_head.weight"  # tied to the input embeddings
        }
        save_file(state, str(path / WEIGHTS_NAME), metadata={"format": "pt"})

    @classmethod
    def from_pretrained(
        cls, path: Union[str, Path], device: Union[str, torch.device] = "cpu"
    ) -> "CellScribe":
        """Load from a local directory or a Hugging Face Hub repo id."""
        from safetensors.torch import load_file

        path = resolve_path(path)
        model = cls(CellScribeConfig.load(path / CONFIG_NAME))
        missing, unexpected = model.load_state_dict(
            load_file(str(path / WEIGHTS_NAME)), strict=False
        )
        if unexpected or set(missing) - {"llm.lm_head.weight"}:
            raise RuntimeError(f"missing={missing} unexpected={unexpected}")
        model.llm.tie_weights()
        return model.to(device).eval()


def resolve_path(path: Union[str, Path]) -> Path:
    if Path(path).is_dir():
        return Path(path)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(path)))
