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
import types
from typing import Any, Optional, Type, TypeVar

import torch
import torch.nn as nn
from transformers import PreTrainedModel


class AdaLNZero(nn.Module):
    """Adaptive layer norm: ``output = norm(x) * (1 + scale) + shift``.

    ``(scale, shift) = ada_proj(conditioning_signal)``, computed in fp32.
    """

    def __init__(self, original_norm_layer: nn.Module, conditioning_dim: int):
        super().__init__()
        self.norm = original_norm_layer
        hidden_size = original_norm_layer.weight.shape[0]
        self.ada_proj = nn.Linear(conditioning_dim, 2 * hidden_size)
        nn.init.constant_(self.ada_proj.weight, 0)
        nn.init.constant_(self.ada_proj.bias, 0)
        # Set by the decoder layer for the duration of its forward pass.
        self.conditioning_signal: Optional[torch.Tensor] = None

    def forward(
        self, x: torch.Tensor, conditioning_signal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if conditioning_signal is None:
            conditioning_signal = self.conditioning_signal
        if conditioning_signal is None:
            return self.norm(x)

        input_dtype = x.dtype
        autocast_device = "cuda" if x.device.type == "cuda" else "cpu"
        # The modulation is computed in fp32 for stability under bf16/fp16
        # autocast; the AdaLN parameters are kept in fp32.
        with torch.amp.autocast(autocast_device, enabled=False):
            normed = self.norm(x.float())
            ada_out = self.ada_proj(conditioning_signal.float())
            if ada_out.dim() == 2:
                # [B, 2H] -> [B, 1, 2H], broadcast over the sequence.
                ada_out = ada_out.unsqueeze(1)

        scale, shift = torch.chunk(ada_out, 2, dim=-1)
        return (normed * (1 + scale) + shift).to(input_dtype)


def _conditioned_forward(
    self, hidden_states, *args, conditioning_signal=None, **kwargs
):
    """The decoder layer's own forward pass, with both norms conditioned."""
    norms = (self.input_layernorm, self.post_attention_layernorm)
    for norm in norms:
        norm.conditioning_signal = conditioning_signal
    try:
        return type(self).forward(self, hidden_states, *args, **kwargs)
    finally:
        for norm in norms:
            norm.conditioning_signal = None


def attach_ada_ln_zero(llm: PreTrainedModel, conditioning_dim: int) -> PreTrainedModel:
    """Model surgery: replace both norms of every decoder layer with AdaLN-Zero."""
    for layer in llm.model.layers:
        for norm_name in ("input_layernorm", "post_attention_layernorm"):
            setattr(
                layer,
                norm_name,
                AdaLNZero(getattr(layer, norm_name), conditioning_dim),
            )
        layer.forward = types.MethodType(_conditioned_forward, layer)
    return llm


T = TypeVar("T", bound=PreTrainedModel)


def create_conditional_model_class(base_model_class: Type[T]) -> Type[T]:
    """Subclass whose ``forward`` officially accepts ``conditioning_signal``, so
    that ``.generate()`` forwards it to every decoder layer."""

    class ConditionalModel(base_model_class):  # type: ignore
        # Explicit signature: ``generate`` inspects it (e.g. to derive
        # ``position_ids`` from the attention mask under left padding).
        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[list[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            conditioning_signal: Optional[torch.Tensor] = None,
            conditioning_default: Optional[torch.Tensor] = None,
            **kwargs,
        ):
            # A per-position plan ([B, T_plan, D], in-context learning) covers the
            # prompt, but under a KV cache each decoding step only forwards the
            # newly generated token(s): slice the plan to the positions being
            # processed so the conditioning stays aligned with the hidden states.
            # A [B, D] signal is broadcast over all positions as is.
            if conditioning_signal is not None and conditioning_signal.dim() == 3:
                conditioning_signal = self._slice_conditioning_plan(
                    plan=conditioning_signal,
                    conditioning_default=conditioning_default,
                    cache_position=kwargs.get("cache_position"),
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                )
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                conditioning_signal=conditioning_signal,
                **kwargs,
            )

        @staticmethod
        def _slice_conditioning_plan(
            plan: torch.Tensor,
            conditioning_default: Optional[torch.Tensor],
            cache_position: Optional[torch.Tensor],
            position_ids: Optional[torch.LongTensor],
            past_key_values: Optional[Any],
        ) -> torch.Tensor:
            """Select the conditioning rows for the positions being processed.

            Positions beyond the plan length (i.e. generated answer tokens) use
            ``conditioning_default`` (the query-cell conditioning) when provided,
            otherwise the last plan row.
            """
            _batch, t_plan, _dim = plan.shape

            # Resolve the absolute positions of the current tokens.
            if cache_position is not None:
                cur = cache_position.to(plan.device)
            elif position_ids is not None:
                cur = position_ids[0].to(plan.device)
            else:
                # Fall back to past length + arange over the current chunk.
                past_len = 0
                if past_key_values is not None:
                    get_len = getattr(past_key_values, "get_seq_length", None)
                    if callable(get_len):
                        past_len = get_len()
                    elif len(past_key_values) > 0:
                        past_len = past_key_values[0][0].shape[-2]
                cur = torch.arange(past_len, past_len + t_plan, device=plan.device)

            idx = cur.clamp(max=t_plan - 1)
            sel = plan[:, idx, :]  # [B, n_cur, D]

            overflow = cur >= t_plan
            if overflow.any() and conditioning_default is not None:
                default = conditioning_default.to(sel.dtype)  # [B, D]
                sel[:, overflow, :] = default.unsqueeze(1)
            return sel

    return ConditionalModel
