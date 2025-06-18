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
import json
import os
from typing import Callable

import haiku as hk
import jax.numpy as jnp
import joblib
from huggingface_hub import hf_hub_download

from multiomics_open_research.common.tokenizer import BinnedOmicTokenizer
from multiomics_open_research.mojo.config import MOJOConfig
from multiomics_open_research.mojo.model import build_mojo_fn

CHECKPOINT_DIRECTORY = "checkpoints/"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"


def _get_dir(model_name: str) -> str:
    """
    Get directory to save files on user machine.
    """
    return os.path.expanduser(
        os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), model_name)
    )


def download_mojo_ckpt() -> tuple[hk.Params, MOJOConfig]:
    """
    Download MOJO checkpoint from Hugging Face.


    Returns:
        Model parameters.
        Model configuration
    """

    save_dir = os.path.join(_get_dir("mojo"), "mojo")

    repo_id = "InstaDeepAI/MOJO"

    # Download parameters
    print("Downloading model's weights...")
    params = joblib.load(
        hf_hub_download(
            repo_id=repo_id,
            filename="jax_params/params.joblib",
            cache_dir=save_dir,
        )
    )

    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="jax_params/config.json",
    )
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = MOJOConfig(**config_dict)

    return params, config


def get_mojo_pretrained_model(
    compute_dtype: jnp.dtype = jnp.float32,
    param_dtype: jnp.dtype = jnp.float32,
    output_dtype: jnp.dtype = jnp.float32,
) -> tuple[hk.Params, Callable, dict[str, BinnedOmicTokenizer], MOJOConfig]:
    """
    Create a Haiku MOJO model.
    model by downloading pre-trained weights and hyperparameters.

    Args:
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

    Returns:
        Model parameters.
        Haiku function to call the model.
        Tokenizer.
        Model config (hyperparameters).

    """
    parameters, config = download_mojo_ckpt()
    tokenizers = {
        omic: BinnedOmicTokenizer(
            n_expressions_bins=config.n_expressions_bins[omic],
            use_max_normalization=config.use_max_normalization[omic],
            normalization_factor=config.normalization_factor[omic],
            prepend_cls_token=False,
            fixed_sequence_length=config.fixed_sequence_length,
            unpadded_length=config.sequence_length,
        )
        for omic in config.n_expressions_bins.keys()
    }

    forward_fn = build_mojo_fn(
        model_config=config,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        output_dtype=output_dtype,
        model_name="multiomics_lm",
    )

    return parameters, forward_fn, tokenizers, config
