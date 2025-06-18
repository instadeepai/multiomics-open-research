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

from typing import Callable

import jax
import jax.numpy as jnp

SUPPORTED_FFN_ACTIVATIONS = ["gelu", "gelu-no-approx", "relu", "swish", "silu", "sin"]


def get_activation_fn(activation_name: str) -> Callable:
    """
    Return activation fn given its name.
    Args:
        activation_name: Activation name.

    Returns:
        activation function.
    """
    if activation_name not in SUPPORTED_FFN_ACTIVATIONS:
        raise NotImplementedError(
            f"Activation {activation_name} not supported yet. "
            f"Supported activations for feed forward "
            f"block are {SUPPORTED_FFN_ACTIVATIONS}"
        )
    if activation_name == "gelu-no-approx":
        activation_fn = lambda x: jax.nn.gelu(x, approximate=False)  # noqa: E731
    elif activation_name == "sin":
        activation_fn = lambda x: jnp.sin(x)  # noqa: E731
    else:
        activation_fn = getattr(jax.nn, activation_name)
    return activation_fn
