# Copyright 2025 InstaDeep Ltd
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

import pathlib
from typing import Callable

import haiku as hk
import jax.numpy as jnp
import joblib

from multiomics_open_research.bulk_rna_bert.config import BulkRNABertConfig
from multiomics_open_research.bulk_rna_bert.model import build_bulk_rna_bert_forward_fn
from multiomics_open_research.common.tokenizer import BinnedOmicTokenizer

CHECKPOINT_DIRECTORY = "checkpoints/"


def get_bulkrnabert_pretrained_model(
    model_name: str,
    compute_dtype: jnp.dtype = jnp.float32,
    param_dtype: jnp.dtype = jnp.float32,
    output_dtype: jnp.dtype = jnp.float32,
    embeddings_layers_to_save: tuple[int, ...] = (),
    checkpoint_directory: str = CHECKPOINT_DIRECTORY,
) -> tuple[hk.Params, Callable, BinnedOmicTokenizer, BulkRNABertConfig]:
    """
    Create a Haiku BulkRNABert model.
    model by downloading pre-trained weights and hyperparameters.

    Args:
        model_name: Name of the model.
        compute_dtype: the type of the activations. fp16 runs faster and is lighter in
            memory. bf16 handles better large int, and is hence more stable ( it avoids
            float overflows ).
        param_dtype: if compute_dtype is fp16, the model weights will be cast to fp16
            during the forward pass anyway. So in inference mode ( not training mode ),
            it is better to use params in fp16 if compute_dtype is fp16 too. During
            training, it is preferable to keep parameters in float32 for better
            numerical stability.
        output_dtype: the output type of the model. it determines the float precision
            of the gradient when training the model.
        embeddings_layers_to_save: Intermediate embeddings to return in the output.
        checkpoint_directory: name of the folder where checkpoints are stored.

    Returns:
        Model parameters.
        Haiku function to call the model.
        Tokenizer.
        Model config (hyperparameters).

    """
    checkpoint_path = pathlib.Path(checkpoint_directory) / model_name

    config = BulkRNABertConfig.parse_file(checkpoint_path / "config.json")
    tokenizer = BinnedOmicTokenizer(
        n_expressions_bins=config.n_expressions_bins,
        use_max_normalization=config.use_max_normalization,
        normalization_factor=config.normalization_factor,
        prepend_cls_token=False,
    )

    config.embeddings_layers_to_save = embeddings_layers_to_save

    forward_fn = build_bulk_rna_bert_forward_fn(
        model_config=config,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        output_dtype=output_dtype,
        model_name="bulk_bert",
    )

    parameters = joblib.load(checkpoint_path / "params.joblib")

    return parameters, forward_fn, tokenizer, config
