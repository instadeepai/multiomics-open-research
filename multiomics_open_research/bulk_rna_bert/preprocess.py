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

import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

MIN_COMMON_GENES = 18_000
DEFAULT_GENE_VERSION = "1"


def get_all_tcga_dataset_files(dataset_path: Path) -> list[Path]:
    sub_folders = [s for s in dataset_path.iterdir() if s.is_dir()]
    files = [next(sub_folder.glob("*.tsv")) for sub_folder in sub_folders]
    return files


def read_tcga_dataframe(df_path: Path) -> pd.DataFrame:
    df = pd.read_csv(df_path, sep="\t", comment="#")
    df = df[~df.gene_id.isna() & ~df.gene_name.isna()]
    return df


def split_gene_id_version(gene_id: str) -> tuple[str, str]:
    gene_id_and_version = gene_id.split(".")
    if len(gene_id_and_version) != 2:
        raise ValueError(
            f"Gene id {gene_id} doesn't have the correct format. Expected format is "
            f"GENE_ID.GENE_VERSION"
        )
    gene_id_no_version, gene_version = gene_id_and_version
    return gene_id_no_version, gene_version


def get_gene_id_from_common_gene_ids(
    gene_ids: list[str],
    reference_gene_ids_no_version: list[str],
    input_with_version: bool,
    output_with_version: bool = True,
) -> list[str]:
    """
    Given a list of gene ids, filters them to only include
    genes that are in a list of reference gene ids

    Args:
        gene_ids: list of gene ids to filter.
        reference_gene_ids_no_version: list of reference gene ids (without version).
        input_with_version: whether input gene ids come with their version.
        output_with_version: if True, filtered gene ids will also carry their version.

    Returns:
        Filtered gene ids using reference genes.
    """
    filtered_gene_ids = []
    reference_gene_ids_no_version_set = set(reference_gene_ids_no_version)
    for gid in gene_ids:
        if input_with_version:
            gene_id, gene_version = split_gene_id_version(gid)
        else:
            gene_id = gid
            gene_version = DEFAULT_GENE_VERSION
        if gene_id in reference_gene_ids_no_version_set and gene_version.isdigit():
            # gene_version.isdigit() allows to remove
            # genes like "ENSG00000002586.20_PAR_Y"
            if output_with_version:
                filtered_gene_ids.append(gid)
            else:
                filtered_gene_ids.append(gene_id)
    return filtered_gene_ids


def preprocess_tcga_rna_seq_dataset(
    dataset_path: Path,
    output_file: Path,
    reference_gene_ids: list[str],
    rna_seq_column: str,
) -> None:
    files = get_all_tcga_dataset_files(dataset_path)
    preprocessed_rows = []
    preprocessed_df_columns = reference_gene_ids + ["identifier"]
    identifiers: dict[str, int] = defaultdict(int)
    for file in files:
        try:
            df = read_tcga_dataframe(file)
            reference_gene_ids_with_versions = get_gene_id_from_common_gene_ids(
                df.gene_id.to_list(),
                reference_gene_ids,
                input_with_version=True,
                output_with_version=True,
            )
            if len(reference_gene_ids_with_versions) < MIN_COMMON_GENES:
                logging.debug(
                    f"file {file} has only "
                    f"{len(reference_gene_ids_with_versions)} common genes id "
                    f"with references genes id"
                )
            else:
                df = df[df.gene_id.isin(reference_gene_ids_with_versions)]
                df.gene_id = df.gene_id.apply(lambda x: split_gene_id_version(x)[0])
                gene_id_diff = set(reference_gene_ids).difference(df.gene_id)
                df_additional = pd.DataFrame(
                    [
                        {"gene_id": additional_gene_id, rna_seq_column: 0.0}
                        for additional_gene_id in gene_id_diff
                    ]
                )
                df = (
                    pd.concat([df, df_additional])
                    .set_index("gene_id")
                    .reindex(reference_gene_ids)
                )
                identifier = file.name.split(".")[0]
                if identifier in identifiers:
                    identifier = identifier + f"-{identifiers[identifier]+1}"
                identifiers[identifier] += 1

                preprocessed_rows.append(df[rna_seq_column].to_list() + [identifier])
        except Exception as e:
            logging.error(f"Error of file {file}: " + str(e))

    df_preprocessed = pd.DataFrame(preprocessed_rows, columns=preprocessed_df_columns)
    df_preprocessed.to_csv(output_file, index=False)
