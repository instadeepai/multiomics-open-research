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

from dataclasses import dataclass
from typing import Optional


@dataclass
class RotaryEmbeddingConfig:
    """
    Parameters to initialize the RotaryEmbedding layer. The rescaling factor allows
    to adapt the rotary embeddings to larger lengths than what was used for training.
    One of this strategy is presented in the Yarn paper: https://arxiv.org/pdf/2309.00071.pdf. # noqa

    Args:

    """

    rescaling_factor: Optional[float]
