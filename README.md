Welcome to this InstaDeep Github repository that gathers the research work done by [Maxence Gélard](https://www.linkedin.com/in/maxence-g%C3%A9lard-015761172/) in the context of his PhD.

### Models 🧬

This repository provides the code and pre-trained weights of three models:

| Model | Data | Description | Framework |
|---|---|---|---|
| [1. BulkRNABert](#1-bulkrnabert) ([paper](https://proceedings.mlr.press/v259/gelard25a.html)) | Bulk RNA-seq | Transformer encoder pre-trained with masked language modeling, whose embeddings are used for cancer-type classification and survival prediction. | JAX |
| [2. MOJO](#2-mojo) ([paper](https://proceedings.mlr.press/v306/gelard26a.html)) | Bulk RNA-seq + DNA methylation | Bimodal model learning joint representations of both modalities for cancer-type classification and survival analysis. | JAX |
| [3. CellScribe](#3-cellscribe) ([paper](https://www.biorxiv.org/content/10.64898/2026.10.07.757352)) | Single-cell RNA-seq | Language model conditioned on single-cell foundation model embeddings that generates a cell's type and its active malignant meta-programs in natural language. | PyTorch |

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

CellScribe is a generative framework that describes a cell along two complementary axes in a single
model: its *identity* (lineage) and its *state* (active transcriptional programs). It conditions a
pretrained language model ([Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
[Qwen Team, 2024](https://arxiv.org/abs/2412.15115)) on frozen embeddings of a single-cell foundation
model ([scFoundation](https://www.nature.com/articles/s41592-024-02305-7), Hao et al., 2024) through
adaptive layer normalization (AdaLN, [Peebles and Xie, 2023](https://arxiv.org/abs/2212.09748)): the
cell embedding sets the scale and shift of every layer normalization of the language model.
Depending on the task-specific prompt, the same model generates:
- **cell typing**: an ontology-grounded cell type (Cell Ontology vocabulary of CELLxGENE), e.g.
  `"The cell type is B cell."`;
- **cell state**: a quantified meta-program profile. Meta-programs are the 41 recurrent malignant
  programs identified across 24 cancer types with the Curated Cancer Cell Atlas (3CA) by
  [Gavish et al., *Hallmarks of transcriptional intratumour heterogeneity across a thousand tumours*,
  Nature 2023](https://www.nature.com/articles/s41586-023-06130-4), and grouped into 11 hallmarks.
  CellScribe lists the active programs with their activity scores and the dominant hallmark, e.g.
  `"Active programs: Cell Cycle - G2/M (1.9), EMT-II (1.1). Dominant hallmark: Cell cycle."`.

The paper also describes reinforcement learning against cell-ontology distance and meta-program
agreement. Beyond these two tasks, CellScribe identifies cell types outside its supervised
vocabulary through in-context learning (see below).

The CellScribe weights are for non-commercial research use only. They include the unmodified
scFoundation encoder weights, which remain under the
[scFoundation Model License](https://github.com/biomap-research/scFoundation/blob/main/MODEL_LICENSE)
of BioMap (see the `LICENSE.md` of the Hugging Face repository for the full terms).

Unlike BulkRNABert and MOJO, CellScribe is implemented in PyTorch. Inputs are raw UMI counts
(not normalised) with HGNC gene symbols as gene names. Genes missing from the input are set to
zero. We provide in `data/cellscribe` two samples of held-out cells: CELLxGENE cells for cell typing
and malignant cells from the Curated Cancer Cell Atlas (3CA) for cell state. To predict their cell
type and cell state, one can use the following code snippet:

```python
import pandas as pd

from multiomics_open_research.cellscribe.pretrained import get_cellscribe_predictor

# Get pretrained CellScribe predictor
predictor = get_cellscribe_predictor(device="cuda")  # or "cpu"

# Cell typing
cells = pd.read_csv("data/cellscribe/cell_typing_sample.csv", index_col=0)
cell_types = predictor.predict_cell_type(cells.values, list(cells.columns))
print(cell_types)
# ['CD4-positive, alpha-beta cytotoxic T cell', 'immature B cell',
#  'CD16-negative, CD56-bright natural killer cell, human', 'CD14-low, CD16-positive monocyte']

# Cell state
cells = pd.read_csv("data/cellscribe/cell_state_sample.csv", index_col=0)
for prediction in predictor.predict_cell_state(cells.values, list(cells.columns)):
    print(prediction.programs, prediction.hallmark)
# [('Stress (in vitro)', 0.9), ('Stress', 0.8), ('Translation initiation', 0.7), ('MYC-targets', 0.6)] Stress or hypoxia
# [('Unfolded protein response', 1.1), ('immunoglobulins (IGs)', 1.0), ('Protein maturation', 0.8), ('Interferon/MHC-II (I)', 0.8), ('Stress', 0.7)] Protein regulation
# [('Cell Cycle - G1/S', 1.8), ('Cell Cycle - G2/M', 1.7), ('Cell cycle (HMG-rich)', 1.4), ('MYC-targets', 0.9), ('EMT-II', 0.8)] Cell cycle
# [('MYC-targets', 1.0), ('Cell Cycle - G2/M', 0.9), ('Cell cycle (HMG-rich)', 0.7), ('Respiration', 0.6)] Oncogenic
```

An `AnnData` object holding raw counts can be used directly with
`predictor.predict_anndata(adata, task="cell_typing")` (or `task="cell_state"`). For CELLxGENE files,
first set `adata.var_names = adata.var["feature_name"]` so that genes are named by their symbols.
Cells are read and processed one batch at a time, so large files can be opened without loading
them in memory with `anndata.read_h5ad(path, backed="r")`.

On GPU, inference runs in bfloat16 by default, as in the paper's evaluation. Pass
`autocast_dtype=None` to any predictor method for float32, the default on CPU.

### In-context learning: cell types outside the training vocabulary

CellScribe can identify cell types it was never trained on through in-context learning. The prompt
lists a few labelled example cells, one line per cell (`"Example cell 1: The cell type is <label>."`),
each line being conditioned by its own cell, and the model infers the type of the query cell.
`data/cellscribe/icl_query_sample.csv` holds a kidney loop of Henle thick ascending limb epithelial
cell, a cell type held out of training. Without examples, CellScribe mistakes it for another segment
of the nephron, the distal convoluted tubule. `data/cellscribe/icl_examples_sample.csv` holds four
labelled example cells (first column `cell_type`): two thick ascending limb cells and two cells of
its closest training type, the thin ascending limb of the loop of Henle.

```python
query = pd.read_csv("data/cellscribe/icl_query_sample.csv", index_col=0)
examples = pd.read_csv("data/cellscribe/icl_examples_sample.csv", index_col=0)
example_labels = examples.pop("cell_type").tolist()

# Without examples: the held-out cell type cannot be predicted
print(predictor.predict_cell_type(query.values, list(query.columns)))
# ['kidney distal convoluted tubule epithelial cell']

# With labelled examples (in-context learning)
print(
    predictor.predict_cell_type_in_context(
        query.values, list(query.columns), examples.values, list(examples.columns), example_labels
    )
)
# ['kidney loop of Henle thick ascending limb epithelial cell']
```

### Using the model directly

To run your own inference loop or to fine-tune CellScribe, one can load the model (a
`torch.nn.Module`) and its components without the predictor:

```python
import pandas as pd
import torch

from multiomics_open_research.cellscribe.pretrained import get_cellscribe_pretrained_model
from multiomics_open_research.cellscribe.prompts import build_in_context_prompt, build_prompt

model, tokenizer, gene_mapper, preprocessor = get_cellscribe_pretrained_model(device="cuda")

# Raw counts -> encoder inputs
cells = pd.read_csv("data/cellscribe/cell_typing_sample.csv", index_col=0)
counts = torch.from_numpy(gene_mapper(cells.values, list(cells.columns))).to("cuda")
scrnaseq_input = preprocessor(counts)

# Inference: generate the answers
prompt = build_prompt(tokenizer, task="cell_typing")  # or task="cell_state"
inputs = tokenizer([prompt] * len(cells), return_tensors="pt").to("cuda")
generated = model.generate(
    inputs.input_ids, inputs.attention_mask, scrnaseq_preprocessed_input=scrnaseq_input
)
print(tokenizer.batch_decode(generated, skip_special_tokens=True))

# In-context learning: one conditioning vector per prompt token
query = pd.read_csv("data/cellscribe/icl_query_sample.csv", index_col=0)
examples = pd.read_csv("data/cellscribe/icl_examples_sample.csv", index_col=0)
example_labels = examples.pop("cell_type").tolist()


def embed(cells: pd.DataFrame) -> torch.Tensor:
    counts = torch.from_numpy(gene_mapper(cells.values, list(cells.columns))).to("cuda")
    return model.embed_cells(preprocessor(counts))  # [n_cells, hidden_size]


with torch.no_grad():
    query_vectors, example_vectors = embed(query), embed(examples)
prompt_ids, spans = build_in_context_prompt(tokenizer, example_labels)
plan = query_vectors[:, None, :].repeat(1, len(prompt_ids), 1)  # the query cell everywhere,
for k, (start, end) in enumerate(spans):
    plan[:, start:end] = example_vectors[k]  # except on each example line
input_ids = torch.tensor([prompt_ids] * len(query), device="cuda")
generated = model.generate(
    input_ids,
    torch.ones_like(input_ids),
    conditioning=plan,
    conditioning_default=query_vectors,  # conditions the generated answer
)
print(tokenizer.batch_decode(generated, skip_special_tokens=True))

# Training: teacher-forced loss on the answers (prompt tokens and padding masked out)
answers = [f"The cell type is {label}.<|im_end|>\n" for label in ["B cell"] * len(cells)]
tokenizer.padding_side = "right"
batch = tokenizer([prompt + answer for answer in answers], return_tensors="pt", padding=True).to("cuda")
labels = batch.input_ids.clone()
labels[:, : len(tokenizer(prompt).input_ids)] = -100
labels[batch.attention_mask == 0] = -100
model.train()
loss = model(
    batch.input_ids, batch.attention_mask, scrnaseq_preprocessed_input=scrnaseq_input, labels=labels
).loss
loss.backward()
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
