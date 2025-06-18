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
from multiomics_open_research.bulk_rna_bert.downstream.config import (
    RNASeqDownStreamConfig,
)
from multiomics_open_research.bulk_rna_bert.downstream.model import (
    RNASeqSurvivalMLP,
    build_bulk_bert_with_head_fn,
)
from multiomics_open_research.bulk_rna_bert.pretrained import CHECKPOINT_DIRECTORY
from multiomics_open_research.common.tokenizer import BinnedOmicTokenizer

MODEL_NAME_TO_HEAD_NAME = {
    "tcga_5_cohorts": "classification_head",
}


def get_pretrained_downstream_model(
    model_name: str,
    compute_dtype: jnp.dtype = jnp.float32,
    param_dtype: jnp.dtype = jnp.float32,
    output_dtype: jnp.dtype = jnp.float32,
    checkpoint_directory: str = CHECKPOINT_DIRECTORY,
) -> tuple[
    hk.Params,
    Callable,
    BinnedOmicTokenizer,
    RNASeqDownStreamConfig,
    BulkRNABertConfig,
]:
    """
    Create a Haiku Nucleotide Transformer
    model by downloading pre-trained weights and hyperparameters.
    Nucleotide Transformer Models have ESM-like architectures.

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
        checkpoint_directory: name of the folder where checkpoints are stored.

    Returns:
        Model parameters.
        Haiku function to call the model.
        Tokenizer.
        Model config (hyperparameters).
        Representation model (LM) config.

    """
    checkpoint_directory_path = pathlib.Path(checkpoint_directory)
    checkpoint_path = checkpoint_directory_path / model_name

    config = RNASeqDownStreamConfig.parse_file(checkpoint_path / "config.json")
    mlm_config = BulkRNABertConfig.parse_file(
        checkpoint_directory_path
        / config.rnaseq_representation_model.checkpoint_path.stem
        / "config.json"
    )
    embeddings_layer_to_use = config.rnaseq_representation_model.embeddings_layer_to_use
    if embeddings_layer_to_use == -1:
        embeddings_layer_to_use = mlm_config.num_layers
    mlm_config.embeddings_layers_to_save = (embeddings_layer_to_use,)

    tokenizer = BinnedOmicTokenizer(
        n_expressions_bins=mlm_config.n_expressions_bins,
        use_max_normalization=mlm_config.use_max_normalization,
        normalization_factor=mlm_config.normalization_factor,
        prepend_cls_token=False,
    )

    if model_name not in MODEL_NAME_TO_HEAD_NAME:
        raise ValueError(f"Model {model_name} not supported.")
    head_fn_name = MODEL_NAME_TO_HEAD_NAME[model_name]

    def head_fn() -> hk.Module:
        model = RNASeqSurvivalMLP(model_config=config.model, name=head_fn_name)
        return model

    forward_fn = build_bulk_bert_with_head_fn(
        model_config=mlm_config,
        head_fn=head_fn,  # type: ignore
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        output_dtype=output_dtype,
        model_name="bulk_bert",
    )

    parameters = joblib.load(checkpoint_path / "params.joblib")

    return parameters, forward_fn, tokenizer, config, mlm_config
