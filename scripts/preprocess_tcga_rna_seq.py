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

import logging
import pathlib

import click

from multiomics_open_research.bulk_rna_bert.preprocess import (
    preprocess_tcga_rna_seq_dataset,
)


@click.command(
    name="preprocess_rnaseq",
    short_help="Preprocess GTEx dataset",
)
@click.option(
    "--dataset-path",
    "dataset_path",
    required=True,
    help="Path to RNASeq dataset",
    type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=pathlib.Path),
)
@click.option(
    "--output-folder",
    "output_folder",
    required=True,
    help="Path to the folder to save the preprocessed dataset",
    type=click.Path(
        exists=False, dir_okay=True, file_okay=True, path_type=pathlib.Path
    ),
)
@click.option(
    "--common-gene-ids-path",
    "common_gene_ids_path",
    required=True,
    help="Path to save the common gene ids",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, path_type=pathlib.Path
    ),
)
@click.option(
    "--rna-seq-column",
    "rna_seq_column",
    required=False,
    help="Name of the RNASeq column in the dataframe to use",
    type=str,
)
def preprocess_rnaseq(
    dataset_path: pathlib.Path,
    output_folder: pathlib.Path,
    common_gene_ids_path: pathlib.Path,
    rna_seq_column: str,
) -> None:
    if not output_folder.is_dir():
        output_folder.mkdir(parents=True)

    with open(common_gene_ids_path, "r") as f:
        reference_gene_ids = [line.strip() for line in f.readlines()]

    output_filename = dataset_path.stem + "_preprocessed.csv"
    output_file = output_folder / output_filename
    preprocess_tcga_rna_seq_dataset(
        dataset_path=dataset_path,
        output_file=output_file,
        reference_gene_ids=reference_gene_ids,
        rna_seq_column=rna_seq_column,
    )


if __name__ == "__main__":
    logger = logging.getLogger("root")
    logger.setLevel(logging.INFO)

    preprocess_rnaseq()
