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

from typing import Optional

import haiku as hk
import jax.numpy as jnp
from haiku import initializers


class SimpleLMHead(hk.Module):
    """
    Basic Language Model head. Transforms final attention block output
    into a distribution over tokens at each sequence position.
    """

    def __init__(
        self,
        embed_dim: int,
        alphabet_size: int,
        add_bias_lm_head: bool = True,
        name: Optional[str] = None,
    ):
        """
        Args:
            embed_dim: Embedding dimension.
            alphabet_size: Number of tokens in the alphabet.
            name: Name of the layer. Defaults to None.
        """
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.alphabet_size = alphabet_size

        # Define layers
        w_init = initializers.VarianceScaling(2.0, "fan_in", "uniform")
        b_init = initializers.VarianceScaling(2.0, "fan_in", "uniform")
        self._final_fc = hk.Linear(
            self.alphabet_size,
            w_init=w_init,
            b_init=b_init,
            with_bias=add_bias_lm_head,
            name="lm_final_fc",
        )

    def __call__(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        # Compute logits
        logits = self._final_fc(x)
        return {"logits": logits}
