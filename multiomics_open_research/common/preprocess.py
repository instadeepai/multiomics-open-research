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

import numpy as np
import pandas as pd

from multiomics_open_research.bulk_rna_bert.config import BulkRNABertConfig
from multiomics_open_research.mojo.config import MOJOConfig


def preprocess_omic(
    omic_df: pd.DataFrame,
    config: BulkRNABertConfig | MOJOConfig,
    omic: str | None = None,
) -> np.ndarray:
    omic_df = omic_df.drop(["identifier", "cohort"], axis=1, errors="ignore")
    omic_array = omic_df.to_numpy()
    if isinstance(config.use_log_normalization, dict):
        assert omic is not None
        use_log_normalization = config.use_log_normalization[omic]
    else:
        use_log_normalization = config.use_log_normalization
    if use_log_normalization:
        omic_array = np.log10(omic_array + 1)
    return omic_array
