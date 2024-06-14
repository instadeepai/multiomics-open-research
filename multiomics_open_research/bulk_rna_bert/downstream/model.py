# Copyright 2024 InstaDeep Ltd
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

from typing import Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import jmp

from multiomics_open_research.bulk_rna_bert.config import BulkRNABertConfig
from multiomics_open_research.bulk_rna_bert.downstream.config import RNASeqMLPConfig
from multiomics_open_research.bulk_rna_bert.model import BulkRNABert


def get_activation_fn(final_activation_str: str) -> Callable:
    if final_activation_str == "identity":
        return lambda x: x
    else:
        return getattr(jax.nn, final_activation_str)


class RNASeqSurvivalMLP(hk.Module):
    def __init__(
        self, model_config: RNASeqMLPConfig, name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)
        self._model_config = model_config
        self._dropout_rate = self._model_config.dropout_rate
        self._mlp = hk.nets.MLP(
            model_config.hidden_sizes,
            activation=get_activation_fn(model_config.mlp_activation),
            activate_final=True,
        )
        self._fc_layer = hk.Linear(model_config.final_layer_n_logits)
        if model_config.layer_norm:
            self._layer_norm = hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True,
            )
        else:
            self._layer_norm = None
        self._final_activation = get_activation_fn(model_config.final_activation)

    def __call__(self, embeddings: jnp.array, is_training: bool = False) -> jnp.array:
        # embeddings' shape: (batch_size, seq_length, emb_dim)
        x = jnp.transpose(
            embeddings, axes=(0, 2, 1)
        )  # shape (batch, emb_dim, seq_length)
        mean_embedding = jnp.mean(x, axis=-1)
        out = self._mlp(mean_embedding) if self._mlp is not None else mean_embedding
        if is_training:
            out = hk.dropout(hk.next_rng_key(), self._dropout_rate, out)
        if self._layer_norm is not None:
            out = self._layer_norm(out)
        out = self._fc_layer(out)
        return self._final_activation(out)


def build_bulk_bert_with_head_fn(
    model_config: BulkRNABertConfig,
    head_fn: Callable[[], Callable[[jnp.ndarray], dict[str, jnp.ndarray]]],
    compute_dtype: jnp.dtype = jnp.float32,
    param_dtype: jnp.dtype = jnp.float32,
    output_dtype: jnp.dtype = jnp.float32,
    model_name: Optional[str] = None,
) -> Callable:
    assert {compute_dtype, param_dtype, output_dtype}.issubset(
        {
            jnp.bfloat16,
            jnp.float32,
            jnp.float16,
        }
    ), f"Please provide a dtype in {jnp.bfloat16, jnp.float32, jnp.float16}"

    policy = jmp.Policy(
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        output_dtype=output_dtype,
    )
    hk.mixed_precision.set_policy(BulkRNABert, policy)

    # Remove it in batch norm to avoid instabilities
    norm_policy = jmp.Policy(
        compute_dtype=jnp.float32,
        param_dtype=param_dtype,
        output_dtype=compute_dtype,
    )
    hk.mixed_precision.set_policy(hk.BatchNorm, norm_policy)
    hk.mixed_precision.set_policy(hk.LayerNorm, norm_policy)
    hk.mixed_precision.set_policy(hk.RMSNorm, norm_policy)

    def forward_fn(
        tokens: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        sequence_mask: Optional[jnp.ndarray] = None,
        is_training: bool = False,
    ):
        """Forward pass"""
        model = BulkRNABert(
            config=model_config,
            name=model_name,
        )
        outs = model(
            tokens=tokens,
            attention_mask=attention_mask,
        )
        embeddings = outs[f"embeddings_{model_config.embeddings_layers_to_save[0]}"]
        head = head_fn()
        head_outs = head(embeddings=embeddings, is_training=is_training)  # type: ignore
        return {"logits": head_outs}

    return forward_fn
