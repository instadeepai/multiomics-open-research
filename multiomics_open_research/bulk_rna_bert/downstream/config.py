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
import pathlib
from typing import Literal, Optional

from pydantic import BaseModel


class RNASeqMLPConfig(BaseModel):
    name: Literal["mlp"]
    hidden_sizes: list[int]
    final_activation: str = "identity"
    use_gene_groups: bool = False
    gene_groups_file_path: Optional[pathlib.Path] = None
    final_layer_n_logits: int = 1
    dropout_rate: float = 0.0
    layer_norm: bool = False
    mlp_activation: str = "selu"
    use_raw_rnaseq: bool = False


class RnaseqRepresentationMLMConfig(BaseModel):
    name: Literal["mlm"]
    checkpoint_path: pathlib.Path
    embeddings_layer_to_use: int


class RNASeqDownStreamConfig(BaseModel):
    rnaseq_representation_model: RnaseqRepresentationMLMConfig
    model: RNASeqMLPConfig
