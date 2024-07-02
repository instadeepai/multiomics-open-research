Welcome to this InstaDeep Github repository that gathers the research work done by [Maxence Gélard](https://www.linkedin.com/in/maxence-g%C3%A9lard-015761172/) in the context of his PhD.

# BulkRNABert

We present BulkRNABert, a transformer-based encoder-only language model pre-trained on bulk RNA-seq data through self-supervision using masked language modeling from
BERT’s method. It achieves state-of-the-art performance in cancer type classification and survival time prediction on TCGA dataset.
In this repository, we provide code to use pre-trained model.

We provide a sample of data in `data/tcga_sample.csv` (from GBMLGG cohort) as well as a text file `common_gene_id.txt` that indicates the gene ids that must be used (and in which order they should appear).

### Get started 🚀

To use the code and pre-trained models, simply:

1. Clone the repository to your local machine.
2. Install the package by running `pip install -e .`.

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
- **bulk_rna_bert_gtex_encode**: BulkRNABert pre-trained on GTEx and ENCODE data
- **bulk_rna_bert_gtex_encode_tcga**: BulkRNABert pre-trained on a mix of GTEx, ENCODE and TCGA data.


### Dataset preprocessing

TCGA dataset has been obtained through the [GDC portal](https://portal.gdc.cancer.gov/).
A sample of raw RNA-seq data is provided in the folder `data/raw_tcga_sample/` as downloaded from the portal. We also provide the preprocessing script (`scripts/preprocess_tcga_rna_seq.py`) that allows you to generate
the preprocessed `data/tcga_sample.csv` file. This script uses the set of genes use by `BulkRNABert` that is provided in `data/common_gene_id.txt`.
To run the preprocessing, one can use:


```
python scripts/preprocess_tcga_rna_seq.py \
--dataset-path data/tcga_sample_gdc/ \
--output-folder data/ \
--common-gene-ids-path data/common_gene_id.txt \
--rna-seq-column tpm_unstranded
```

### Downstream task example

A example notebook `examples/downstream_task_example.ipynb` illustrates an inference with the classification model trained on the 5 cohorts (BRCA, BLCA, GBMLGG, LUAD, UCEC) classification problem.

## Citing our work 📚

If you find this repository useful in your work, please add a citation to our associated paper:

[BulkRNABert paper](https://doi.org/10.1101/2024.06.18.599483):

```bibtex
@article{gelard2024bulkrnabert,
  title={BulkRNABert: Cancer prognosis from bulk RNA-seq based language models},
  author={Gelard, Maxence and Richard, Guillaume and Pierrot, Thomas and Cournede, Paul-Henry},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.06.18.599483},
  publisher={Cold Spring Harbor Laboratory},
}
```
