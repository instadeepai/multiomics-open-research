Welcome to this InstaDeep Github repository that gathers the research work done by [Maxence Gélard](https://www.linkedin.com/in/maxence-g%C3%A9lard-015761172/) in the context of his PhD.

### Models 🧬

This repository provides the code and pre-trained weights of three models:

| Model | Data | Description | Framework |
|---|---|---|---|
| [1. BulkRNABert](#1-bulkrnabert) ([paper](https://proceedings.mlr.press/v259/gelard25a.html)) | Bulk RNA-seq | Transformer encoder pre-trained with masked language modeling, whose embeddings are used for cancer-type classification and survival prediction. | JAX |
| [2. MOJO](#2-mojo) ([paper](https://proceedings.mlr.press/v306/gelard26a.html)) | Bulk RNA-seq + DNA methylation | Bimodal model learning joint representations of both modalities for cancer-type classification and survival analysis. | JAX |
| [3. CellScribe](#3-cellscribe) ([paper](https://www.biorxiv.org/content/10.64898/2026.10.07.757352)) | Single-cell RNA-seq | Language model conditioned on single-cell foundation model embeddings that generates a cell's type and cell states in natural language. | PyTorch |

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

from multiomics_open_research.bulk_rna_bert.pretrained import get_bulkrnabert_pretrained_model
from multiomics_open_research.common.preprocess import preprocess_omic

# Get pretrained model
parameters, forward_fn, tokenizer, config = get_bulkrnabert_pretrained_model(
    model_name="bulk_rna_bert_tcga",
    embeddings_layers_to_save=(4,),
    checkpoint_directory="checkpoints/",
)
forward_fn = hk.transform(forward_fn)

# Get bulk RNASeq data and tokenize it
gene_expression_df = pd.read_csv("data/bulkrnabert/tcga_sample.csv")
gene_expression_array = preprocess_omic(gene_expression_df, config)
gene_expression_ids = tokenizer.batch_tokenize(gene_expression_array)
gene_expression_ids = jnp.asarray(gene_expression_ids, dtype=jnp.int32)

# Inference
random_key = jax.random.PRNGKey(0)
outs = forward_fn.apply(parameters, random_key, gene_expression_ids)

# Get mean embeddings from layer 4
gene_expression_mean_embeddings = outs["embeddings_4"].mean(axis=1)
```
Supported model names are:
- **bulk_rna_bert_tcga**: BulkRNABert pre-trained on TCGA data.
- **bulk_rna_bert_gtex_encode**: BulkRNABert pre-trained on GTEx and ENCODE data
- **bulk_rna_bert_gtex_encode_tcga**: BulkRNABert pre-trained on a mix of GTEx, ENCODE and TCGA data.


### Dataset preprocessing

TCGA dataset has been obtained through the [GDC portal](https://portal.gdc.cancer.gov/).
A sample of raw RNA-seq data is provided in the folder `data/bulkrnabert/tcga_sample_gdc/` as downloaded from the portal. We also provide the preprocessing script (`scripts/preprocess_tcga_rna_seq.py`) that turns it into a preprocessed file,
written as `tcga_sample_gdc_preprocessed.csv` in the output folder. This script uses the set of genes used by `BulkRNABert` that is provided in `data/bulkrnabert/common_gene_id.txt`.
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

* 📜 **[Read the Paper (ICML 2026)](https://proceedings.mlr.press/v306/gelard26a.html)**
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

# 3. CellScribe

* 📜 **[Read the Paper (bioRxiv 2026)](https://www.biorxiv.org/content/10.64898/2026.10.07.757352)**
* 🤗 **[Hugging Face Link](https://huggingface.co/InstaDeepAI/CellScribe)**

CellScribe is a generative framework that describes a cell along two complementary axes in a
single model: its *identity* (lineage) and its *state* (active transcriptional programs). It
conditions a pretrained language model
([Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)) on frozen embeddings
of a single-cell foundation model ([scFoundation](https://www.nature.com/articles/s41592-024-02305-7))
through adaptive layer normalization (AdaLN): the cell embedding sets the scale and shift of every
layer normalization of the language model. Depending on the task-specific prompt, the same model
generates:

- **Cell typing**: an ontology-grounded cell type (Cell Ontology vocabulary of CELLxGENE).
  Example: `"The cell type is B cell."`
- **Cell state**: a quantified meta-program profile. Meta-programs are the 41 recurrent
  malignant programs identified across 24 cancer types with the Curated Cancer Cell Atlas (3CA)
  by [Gavish et al., *Hallmarks of transcriptional intratumour heterogeneity across a thousand tumours*, Nature 2023](https://www.nature.com/articles/s41586-023-06130-4), and
  grouped into 11 hallmarks. CellScribe lists the active programs with their activity scores
  and the dominant hallmark.
  Example: `"Active programs: Cell Cycle - G2/M (1.9), EMT-II (1.1). Dominant hallmark: Cell cycle."`

The paper also describes reinforcement learning against cell-ontology distance and
meta-program agreement. Beyond these two tasks, CellScribe identifies cell types outside its
supervised vocabulary through in-context learning: the prompt lists a few labelled example
cells, each conditioning its own line, and the model infers the type of the query cell.

## Model

- **Cell encoder**: [scFoundation](https://www.nature.com/articles/s41592-024-02305-7)
  (Hao et al., 2024). The encoder embeds the cell's raw counts over 19,264 genes.
- **Conditioning**: a linear projection followed by a layer normalization turns the embedding
  into a conditioning vector. Through adaptive layer normalization (AdaLN,
  [Peebles and Xie, 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html)), that vector sets the scale and
  shift of every layer normalization of the language model. The AdaLN projections are
  initialised to zero (AdaLN-Zero), so training starts from the unmodified language model.
- **Language model**: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
  ([Qwen Team, 2024](https://qwenlm.github.io/blog/qwen2.5/)), prompted with a task-specific system
  prompt and question.

The model was trained on [CELLxGENE Census](https://chanzuckerberg.github.io/cellxgene-census/)
datasets (cell typing) and on the [Curated Cancer Cell Atlas, 3CA](https://www.weizmann.ac.il/sites/3CA/)
(cell typing and cell state).

## Usage

Unlike BulkRNABert and MOJO, CellScribe is implemented in PyTorch. Its weights are downloaded
from Hugging Face. The examples below use the example cells of `data/cellscribe` (raw counts
with gene symbols as columns), all held out of training:

- `cell_typing_sample.csv`: four CELLxGENE immune cells, for cell typing.
- `cell_state_sample.csv`: four malignant cells from the Curated Cancer Cell Atlas (3CA),
  for cell state.
- `icl_query_sample.csv` and `icl_examples_sample.csv`, for in-context learning: a query
  cell of a cell type held out of training, and labelled example cells (first column
  `cell_type`).

### Using the model directly

The model is a `torch.nn.Module`, which can be used in a custom inference loop or fine-tuned.

#### Loading

```python
import pandas as pd
import torch

from multiomics_open_research.cellscribe.pretrained import get_cellscribe_pretrained_model

model, tokenizer = get_cellscribe_pretrained_model(device="cuda")  # or "cpu"


def load_example(name: str) -> pd.DataFrame:
    return pd.read_csv(f"data/cellscribe/{name}.csv", index_col=0)
```

#### Cell typing

```python
cells = load_example("cell_typing_sample")
scrnaseq_input = model.preprocess(cells.values, list(cells.columns))  # raw counts -> encoder inputs
prompt = model.build_prompt(tokenizer, task="cell_typing")
inputs = tokenizer([prompt] * len(cells), return_tensors="pt").to(model.device)
generated = model.generate(
    inputs.input_ids, inputs.attention_mask, scrnaseq_preprocessed_input=scrnaseq_input
)
answers = tokenizer.batch_decode(generated, skip_special_tokens=True)
print([model.parse_cell_type(answer) for answer in answers])
# ['CD4-positive, alpha-beta cytotoxic T cell', 'immature B cell',
#  'CD16-negative, CD56-bright natural killer cell, human', 'CD14-low, CD16-positive monocyte']
```

#### Cell state

```python
cells = load_example("cell_state_sample")
scrnaseq_input = model.preprocess(cells.values, list(cells.columns))
prompt = model.build_prompt(tokenizer, task="cell_state")
inputs = tokenizer([prompt] * len(cells), return_tensors="pt").to(model.device)
generated = model.generate(
    inputs.input_ids, inputs.attention_mask, scrnaseq_preprocessed_input=scrnaseq_input
)
for answer in tokenizer.batch_decode(generated, skip_special_tokens=True):
    programs, hallmark = model.parse_cell_state(answer)
    print(programs, hallmark)
# [('Stress (in vitro)', 0.9), ('Stress', 0.8), ('Translation initiation', 0.7), ('MYC-targets', 0.6)] Stress or hypoxia
# [('Unfolded protein response', 1.1), ('immunoglobulins (IGs)', 1.0), ('Protein maturation', 0.8), ('Interferon/MHC-II (I)', 0.8), ('Stress', 0.7)] Protein regulation
# [('Cell Cycle - G1/S', 1.8), ('Cell Cycle - G2/M', 1.7), ('Cell cycle (HMG-rich)', 1.4), ('MYC-targets', 0.9), ('EMT-II', 0.8)] Cell cycle
# [('MYC-targets', 1.0), ('Cell Cycle - G2/M', 0.9), ('Cell cycle (HMG-rich)', 0.7), ('Respiration', 0.6)] Oncogenic
```

#### In-context learning

Each example line of the prompt is conditioned by its own cell, the rest of the prompt and
the answer by the query cell.

```python
query = load_example("icl_query_sample")
examples = load_example("icl_examples_sample")
example_labels = examples.pop("cell_type").tolist()

with torch.no_grad():
    query_vectors = model.embed_cells(model.preprocess(query.values, list(query.columns)))
    example_vectors = model.embed_cells(model.preprocess(examples.values, list(examples.columns)))
prompt_ids, spans = model.build_in_context_prompt(tokenizer, example_labels)
plan = query_vectors.expand(len(prompt_ids), -1).clone()  # the query cell on every token,
for k, (start, end) in enumerate(spans):
    plan[start:end] = example_vectors[k]  # except the example lines
input_ids = torch.tensor([prompt_ids], device=model.device)
generated = model.generate(
    input_ids,
    torch.ones_like(input_ids),
    conditioning=plan[None],  # batch of one query
    conditioning_default=query_vectors,  # conditions the generated answer
)
print(tokenizer.batch_decode(generated, skip_special_tokens=True))
# ['The cell type is kidney loop of Henle thick ascending limb epithelial cell.']
```

#### Training

A teacher-forced loss on the answers (prompt tokens and padding masked out):

```python
cells = load_example("cell_typing_sample")
cell_types = [
    "CD4-positive, alpha-beta cytotoxic T cell",
    "immature B cell",
    "CD16-negative, CD56-bright natural killer cell, human",
    "CD14-low, CD16-positive monocyte",
]
scrnaseq_input = model.preprocess(cells.values, list(cells.columns))
prompt = model.build_prompt(tokenizer, task="cell_typing")
answers = [f"The cell type is {cell_type}.<|im_end|>\n" for cell_type in cell_types]
tokenizer.padding_side = "right"
batch = tokenizer([prompt + answer for answer in answers], return_tensors="pt", padding=True)
batch = batch.to(model.device)
labels = batch.input_ids.clone()
labels[:, : len(tokenizer(prompt).input_ids)] = -100
labels[batch.attention_mask == 0] = -100
model.train()
loss = model(
    batch.input_ids, batch.attention_mask, scrnaseq_preprocessed_input=scrnaseq_input, labels=labels
).loss
loss.backward()
```

### Using CellScribePredictor

`CellScribePredictor` wraps the model, tokenizer and preprocessing: raw counts in, parsed
answers out. It runs in float32 by default (pass `autocast_dtype="auto"` to any method for
bfloat16 on GPU, as in the paper's evaluation).

```python
predictor = model.predictor(tokenizer)  # model and tokenizer loaded as above
```

#### Cell typing

```python
cells = load_example("cell_typing_sample")
print(predictor.predict_cell_type(cells.values, list(cells.columns)))
# ['CD4-positive, alpha-beta cytotoxic T cell', 'immature B cell',
#  'CD16-negative, CD56-bright natural killer cell, human', 'CD14-low, CD16-positive monocyte']
```

#### Cell state

```python
cells = load_example("cell_state_sample")
for prediction in predictor.predict_cell_state(cells.values, list(cells.columns)):
    print(prediction.programs, prediction.hallmark)
# [('Stress (in vitro)', 0.9), ('Stress', 0.8), ('Translation initiation', 0.7), ('MYC-targets', 0.6)] Stress or hypoxia
# [('Unfolded protein response', 1.1), ('immunoglobulins (IGs)', 1.0), ('Protein maturation', 0.8), ('Interferon/MHC-II (I)', 0.8), ('Stress', 0.7)] Protein regulation
# [('Cell Cycle - G1/S', 1.8), ('Cell Cycle - G2/M', 1.7), ('Cell cycle (HMG-rich)', 1.4), ('MYC-targets', 0.9), ('EMT-II', 0.8)] Cell cycle
# [('MYC-targets', 1.0), ('Cell Cycle - G2/M', 0.9), ('Cell cycle (HMG-rich)', 0.7), ('Respiration', 0.6)] Oncogenic
```

An `AnnData` object holding raw counts can be used directly with
`predictor.predict_anndata(adata, task="cell_typing")` (or `task="cell_state"`).

#### In-context learning

```python
query = load_example("icl_query_sample")
examples = load_example("icl_examples_sample")
example_labels = examples.pop("cell_type").tolist()

# Without examples: the held-out cell type cannot be predicted
print(predictor.predict_cell_type(query.values, list(query.columns)))
# ['kidney distal convoluted tubule epithelial cell']

# With labelled examples
print(
    predictor.predict_cell_type_in_context(
        query.values, list(query.columns), examples.values, list(examples.columns), example_labels
    )
)
# ['kidney loop of Henle thick ascending limb epithelial cell']
```

## Input

- **Counts**: raw UMI counts, not normalised values. The model applies its own normalisation
  (CP10k + log1p).
- **Gene names**: HGNC symbols. For CELLxGENE files, set
  `adata.var_names = adata.var["feature_name"]`.
- **Genes**: the cell encoder reads a fixed list of 19,264 genes (`gene_list.txt`). Datasets
  rarely measure exactly these genes, so `model.preprocess` maps the input genes onto this
  list: listed genes absent from the input are given a count of zero, and genes outside the
  list are dropped.

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

[MOJO](https://proceedings.mlr.press/v306/gelard26a.html):

```bibtex
@InProceedings{pmlr-v306-gelard26a,
  title =      {Bimodal masked language modeling for bulk {RNA}-seq and {DNA} methylation representation learning},
  author =       {G\'{e}lard, Maxence and Benkirane, Hakim and Pierrot, Thomas and Richard, Guillaume and Courn\`{e}de, Paul-Henry},
  booktitle =      {Proceedings of the 43rd International Conference on Machine Learning},
  pages =      {34463--34495},
  year =      {2026},
  editor =      {Zhang, Tong and Dudik, Miroslav and Jaggi, Martin and Agarwal, Alekh and Li, Sharon and Schuurmans, Dale and Zhu, Jerry and Berkenkamp, Felix and Dong, Hanze and Bietti, Alberto},
  volume =      {306},
  series =      {Proceedings of Machine Learning Research},
  month =      {06--11 Jul},
  publisher =    {PMLR},
  pdf =      {https://raw.githubusercontent.com/mlresearch/v306/main/assets/gelard26a/gelard26a.pdf},
  url =      {https://proceedings.mlr.press/v306/gelard26a.html},
}
```

[CellScribe](https://www.biorxiv.org/content/10.64898/2026.10.07.757352):

```bibtex
@article{gelard2026closed,
  title={From Closed-Set Annotation to In-Context Discovery: Biologically Grounded LLMs for Cellular Profiling},
  author={Gelard, Maxence and Benkirane, Hakim and Pierrot, Thomas and Cournede, Paul-Henry and P. de Almeida, Bernardo},
  journal={bioRxiv},
  pages={2026--10},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```

## License 📜

The code and model weights in this repository are released by InstaDeep Ltd and
CentraleSupélec under the [Apache License 2.0](LICENSE.md). The sample data in `data/` are
from [TCGA](https://portal.gdc.cancer.gov/) (GDC), [CELLxGENE](https://cellxgene.cziscience.com/)
and the [Curated Cancer Cell Atlas (3CA)](https://www.weizmann.ac.il/sites/3CA/) and remain under
their sources' terms. The CellScribe weights, downloaded from
[Hugging Face](https://huggingface.co/InstaDeepAI/CellScribe), include the scFoundation encoder
weights, which remain under the
[scFoundation Model License](https://github.com/biomap-research/scFoundation/blob/main/MODEL_LICENSE)
(non-commercial research use only).
