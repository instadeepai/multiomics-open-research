Welcome to this InstaDeep Github repository that gathers the research work done by [Maxence Gélard](https://www.linkedin.com/in/maxence-g%C3%A9lard-015761172/) in the context of his PhD.

### Get started 🚀

To use the code and pre-trained models, simply:

1. Clone the repository to your local machine.
2. Install the package by running `pip install -e .`.


# 1. BulkRNABert

* 📜 **[Read the Paper (Machine Learning for Health 2024)](https://proceedings.mlr.press/v259/gelard25a.html)**
* 🤗 **[Hugging Face Link](https://huggingface.co/InstaDeepAI/BulkRNABert)**

We present BulkRNABert, a transformer-based encoder-only language model pre-trained on bulk RNA-seq data through self-supervision using masked language modeling from
BERT’s method. It achieves state-of-the-art performance in cancer type classification and survival time prediction on TCGA dataset.
In this repository, we provide code to use pre-trained model.

We provide a sample of data in `data/bulkrnabert/tcga_sample.csv` (from the TCGA GBM/LGG cohort) as well as a text file `common_gene_id.txt` that indicates the gene ids that must be used (and in which order they should appear).

You can then do the inference using:
```python
import haiku as hk
import jax
import jax.numpy as jnp
import pandas as pd

from multiomics_open_research.bulk_rna_bert.pretrained import get_pretrained_model
from multiomics_open_research.common.preprocess import preprocess_omic

# Get pretrained model
parameters, forward_fn, tokenizer, config = get_pretrained_model(
    model_name="bulk_rna_bert_tcga",
    embeddings_layers_to_save=(4,),
    checkpoint_directory="checkpoints/",
)
forward_fn = hk.transform(forward_fn)

# Get bulk RNASeq data and tokenize it
rna_seq_df = pd.read_csv("data/bulkrnabert/tcga_sample.csv")
rna_seq_array = preprocess_omic(rna_seq_df, config)
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
the preprocessed `data/bulkrnabert/tcga_sample.csv` file. This script uses the set of genes use by `BulkRNABert` that is provided in `data/common_gene_id.txt`.
To run the preprocessing, one can use:


```
python scripts/preprocess_tcga_rna_seq.py \
--dataset-path data/bulkrnabert/tcga_sample_gdc/ \
--output-folder data/bulkrnabert/ \
--common-gene-ids-path data/bulkrnabert/common_gene_id.txt \
--rna-seq-column tpm_unstranded
```

### Downstream task example

An example notebook `examples/downstream_task_example.ipynb` illustrates an inference with the classification model trained on the 5 cohorts (BRCA, BLCA, GBMLGG, LUAD, UCEC) classification problem.

# 2. MOJO

* 📜 **[Read the Paper (ICML Workshop on Generative AI and Biology 2025)]()**
* 🤗 **[Hugging Face Link](https://huggingface.co/InstaDeepAI/MOJO)**

MOJO is a model that learns joint representations of bulk RNA-seq and DNA methylation tailored for cancer-type classification and survival analysis.

We provide in `data/mojo` samples of bulk RNA-seq and DNA Methylation data extracted from TCGA dataset.
To get MOJO's bimodal embedding for pairs of bulk RNA-seq and DNA methylation one can use the following code snippet:

```python
import haiku as hk
import jax
import jax.numpy as jnp
import pandas as pd

from multiomics_open_research.mojo.pretrained import get_mojo_pretrained_model
from multiomics_open_research.common.preprocess import preprocess_omic

# Get pretrained MOJO model
parameters, forward_fn, tokenizers, config = get_mojo_pretrained_model()
forward_fn = hk.transform(forward_fn)

# Get bulk RNASeq and Methylation data and tokenize them
omic_dfs = {
    "rnaseq": pd.read_csv("data/mojo/tcga_rnaseq_sample.csv"),
    "methylation": pd.read_csv("data/mojo/tcga_methylation_sample.csv")
}
omic_arrays = {
    omic: preprocess_omic(df, config, omic)
    for omic, df in omic_dfs.items()
}
tokens_ids = {
    omic: jnp.asarray(tokenizers[omic].batch_tokenize(omic_array, pad_to_fixed_length=True), dtype=jnp.int32)
    for omic, omic_array in omic_arrays.items()
}

# Inference
random_key = jax.random.PRNGKey(0)
outs = forward_fn.apply(parameters, random_key, tokens_ids)

# Get embedding from last transformer layer
mean_embedding = outs["after_transformer_embedding"].mean(axis=1)
```

## Citing our work 📚

If you find this repository useful in your work, please add a citation to our associated papers:

[BulkRNABert](https://proceedings.mlr.press/v259/gelard25a.html):

```bibtex
@InProceedings{pmlr-v259-gelard25a,
  title = 	 {BulkRNABert: Cancer prognosis from bulk RNA-seq based language models},
  author =       {G{\'{e}}lard, Maxence and Richard, Guillaume and Pierrot, Thomas and Courn{\`{e}}de, Paul-Henry},
  booktitle = 	 {Proceedings of the 4th Machine Learning for Health Symposium},
  pages = 	 {384--400},
  year = 	 {2025},
  editor = 	 {Hegselmann, Stefan and Zhou, Helen and Healey, Elizabeth and Chang, Trenton and Ellington, Caleb and Mhasawade, Vishwali and Tonekaboni, Sana and Argaw, Peniel and Zhang, Haoran},
  volume = 	 {259},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {15--16 Dec},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v259/gelard25a.html},
}
```

[MOJO]():

```bibtex
```
