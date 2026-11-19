# Copyright 2026 InstaDeep Ltd and CentraleSupélec
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Union

import torch
from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizerBase

from multiomics_open_research.cellscribe.inference import CellScribePredictor
from multiomics_open_research.cellscribe.modeling_cellscribe import CellScribe

ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"
REPO_ID = "InstaDeepAI/CellScribe"


def _get_dir(model_name: str) -> str:
    """
    Get directory to save files on user machine.
    """
    return os.path.expanduser(
        os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), model_name)
    )


def download_cellscribe_ckpt() -> str:
    """
    Download CellScribe checkpoint from Hugging Face.

    Returns:
        Local directory holding the weights, configuration, tokenizer, gene list
        and cell-type vocabulary.
    """
    print("Downloading model's weights...")
    return snapshot_download(
        repo_id=REPO_ID, cache_dir=os.path.join(_get_dir("cellscribe"), "cellscribe")
    )


def get_cellscribe_predictor(
    device: Union[str, torch.device] = "cpu",
) -> CellScribePredictor:
    """
    High-level predictor: raw counts in, cell types / cell states out.

    Args:
        device: Device to run the model on, e.g. "cuda".

    Returns:
        Predictor exposing ``predict_cell_type`` and ``predict_cell_state``.
    """
    return CellScribePredictor.from_pretrained(
        download_cellscribe_ckpt(), device=device
    )


def get_cellscribe_pretrained_model(
    device: Union[str, torch.device] = "cpu",
) -> tuple[CellScribe, PreTrainedTokenizerBase]:
    """
    Pretrained CellScribe model and its tokenizer, without the high-level
    predictor (e.g. for custom inference or fine-tuning).

    Args:
        device: Device to put the model on, e.g. "cuda".

    Returns:
        model: ``torch.nn.Module`` (in eval mode). ``model.preprocess(counts,
            gene_names)`` turns raw counts into the encoder inputs;
            ``model(input_ids, attention_mask, scrnaseq_preprocessed_input=...,
            labels=...)`` returns the language-model outputs (logits, and the loss
            when ``labels`` is given); ``model.generate(...)`` returns the
            generated tokens; ``model.predictor(tokenizer)`` gives the high-level
            predictor.
        tokenizer: Tokenizer of the language model (left padding).
    """
    from transformers import AutoTokenizer

    path = download_cellscribe_ckpt()
    model = CellScribe.from_pretrained(path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.padding_side = "left"
    return model, tokenizer
