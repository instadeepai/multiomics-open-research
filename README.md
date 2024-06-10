Welcome to this InstaDeep Github repository that gathers the research work done by [Maxence Gélard](https://www.linkedin.com/in/maxence-g%C3%A9lard-015761172/) in the context of his PhD.

# BulkRNABert

In this BulkRNABert a transformer-based encoder-only language model pre-trained on bulk RNA-seq data through self-supervision using masked language modeling from
BERT’s method. It achieves state-of-the-art performance in cancer type classification and survival time prediction on TCGA dataset.
In this repository, we provide code to use pre-trained model.

We provide a sample of data in `data/tcga_sample.csv` to indicate the gene ids that must be used (and in which order they should appear).

#### Get started 🚀

To use the code and pre-trained models, simply:

1. Clone the repository to your local machine.
2. Install the package by running `pip install .`.

You can then do the inference using:
```python
import haiku as hk
import jax
import jax.numpy as jnp
import pandas as pd

from multiomics_open_research.bulk_rna_bert.pretrained import get_pretrained_model
from multiomics_open_research.bulk_rna_bert.preprocess import preprocess_rna_seq_for_bulkrnabert

# Get pretrained model
parameters, forward_fn, tokenizer, config = get_pretrained_model(
    model_name="bulk_rna_bert_tcga",
    embeddings_layers_to_save=(4,),
    checkpoint_directory="checkpoints/",
)
forward_fn = hk.transform(forward_fn)

# Get bulk RNASeq data and tokenize it
rna_seq_df = pd.read_csv("data/tcga_sample.csv")
rna_seq_array = preprocess_rna_seq_for_bulkrnabert(rna_seq_df, config)
tokens_ids = tokenizer.batch_tokenize(rna_seq_array)
tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)

# Inference
random_key = jax.random.PRNGKey(0)
outs = forward_fn.apply(parameters, random_key, tokens)

# Get mean embeddings from layer 4
mean_embedding = outs["embeddings_4"].mean(axis=1)
```
Supported model names are:
- **bulk_rna_bert_tcga**: BulkRNABert pre-trained on TCGA data.