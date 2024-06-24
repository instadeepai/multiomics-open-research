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

from typing import List

import numpy as np


class BinnedExpressionTokenizer:
    """
    Tokenizer that bins gene expressions to convert them to tokens.
    """

    def __init__(
        self,
        n_expressions_bins: int,
        use_max_normalization: bool = True,
        normalization_factor: float = 1.0,
        prepend_cls_token: bool = False,
    ):
        self._n_expressions_bins = n_expressions_bins
        self._use_max_normalization = use_max_normalization
        self._normalization_factor = normalization_factor
        self._prepend_cls_token = prepend_cls_token

        if self._use_max_normalization:
            self._gene_expression_bins = np.linspace(0.0, 1.0, self._n_expressions_bins)
        else:
            self._gene_expression_bins = np.linspace(
                0.0, normalization_factor, self._n_expressions_bins
            )

        standard_tokens = list(map(str, range(len(self._gene_expression_bins))))
        self._pad_token = "<pad>"
        self._mask_token = "<mask>"
        self._class_token = "<cls>"
        self._unk_token = "<unk>"
        self._eos_token = "<eos>"
        self._bos_token = "<bos>"

        special_tokens = [
            self._pad_token,
            self._mask_token,
            self._class_token,
            self._unk_token,
            self._eos_token,
            self._bos_token,
        ]

        self._all_tokens = standard_tokens + special_tokens
        self._standard_tokens = standard_tokens
        self._special_tokens = special_tokens

        self._tokens_to_ids = {tok: i for i, tok in enumerate(self._all_tokens)}
        self._ids_to_tokens = {i: tok for tok, i in self._tokens_to_ids.items()}

    @property
    def gene_expression_bins(self) -> np.ndarray:
        return self._gene_expression_bins

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @property
    def mask_token(self) -> str:
        return self._mask_token

    @property
    def class_token(self) -> str:
        return self._class_token

    @property
    def unk_token(self) -> str:
        return self.unk_token

    @property
    def eos_token(self) -> str:
        return self._eos_token

    @property
    def bos_token(self) -> str:
        return self._bos_token

    @property
    def pad_id(self) -> int:
        return self.token_to_id(self.pad_token)

    @property
    def mask_id(self) -> int:
        return self.token_to_id(self.mask_token)

    @property
    def class_id(self) -> int:
        return self.token_to_id(self.class_token)

    @property
    def vocabulary(self) -> List[str]:
        return self._all_tokens

    @property
    def standard_tokens(self) -> List[str]:
        return self._standard_tokens

    @property
    def special_tokens(self) -> List[str]:
        return self._special_tokens

    def id_to_token(self, token_id: int) -> str:
        try:
            return self._ids_to_tokens[token_id]
        except KeyError:
            raise KeyError(f"Token id {token_id} not found in vocabulary")

    def token_to_id(self, token: str) -> int:
        try:
            return self._tokens_to_ids[token]
        except KeyError:
            raise KeyError(f"Token {token} not found in vocabulary")

    def tokenize(self, gene_expressions: np.ndarray) -> np.ndarray:
        """
        Tokenize a gene expression array and return an array of bin ids.

        Args:
            gene_expressions: Gene expressions sequence to be tokenized.

        Returns:
            List of tokens ids.
        """
        if self._use_max_normalization:
            gene_expressions /= self._normalization_factor
        tokens_ids = np.digitize(gene_expressions, self._gene_expression_bins)
        tokens_ids[gene_expressions == 0.0] = 0
        if self._prepend_cls_token:
            tokens_ids = np.concatenate([[self.class_id], tokens_ids])
        return tokens_ids

    def batch_tokenize(self, gene_expressions: np.ndarray) -> np.ndarray:
        """
        Tokenizes a batch of gene expressions.

        Args:
            gene_expressions: gene expressions sequence to be tokenized.

        Returns:
            Tokenized gene expressions.
        """
        return np.vstack([self.tokenize(g) for g in gene_expressions])
