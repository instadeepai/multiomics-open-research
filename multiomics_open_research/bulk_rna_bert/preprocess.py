import numpy as np
import pandas as pd

from multiomics_open_research.bulk_rna_bert.config import BulkRNABertConfig


def preprocess_rna_seq_for_bulkrnabert(
    rna_seq_df: pd.DataFrame, config: BulkRNABertConfig
) -> np.ndarray:
    rna_seq_array = rna_seq_df.to_numpy()
    if config.use_log_normalization:
        rna_seq_array = np.log10(rna_seq_array + 1)
    if config.use_max_normalization:
        rna_seq_array /= config.normalization_factor
    return rna_seq_array
