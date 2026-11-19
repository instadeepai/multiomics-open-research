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
"""Task prompts, exactly as used in training (AdaLN variant: no cell token)."""

from __future__ import annotations

from typing import Sequence

TASK_CELL_TYPING = "cell_typing"
TASK_CELL_STATE = "cell_state"
ANSWER_PREFIX = "The cell type is "

PROMPTS: dict[str, dict[str, str]] = {
    TASK_CELL_TYPING: {
        "system": (
            "You are a helpful assistant who answers questions "
            "about cell-types using single-cell RNA-seq data."
        ),
        "user": "What is the cell type in this single-cell RNA-seq sample?",
    },
    TASK_CELL_STATE: {
        "system": (
            "You are a helpful assistant who identifies active "
            "transcriptional meta-programs and their activation "
            "levels from single-cell RNA-seq data."
        ),
        "user": "What are the active meta-programs in this single-cell RNA-seq sample?",
    },
}


def build_prompt(tokenizer, task: str) -> str:
    """Chat-formatted prompt text, ending at the assistant-turn opening."""
    if task not in PROMPTS:
        raise ValueError(f"unknown task {task!r}; choose from {sorted(PROMPTS)}")
    messages = [
        {"role": "system", "content": PROMPTS[task]["system"]},
        {"role": "user", "content": PROMPTS[task]["user"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_in_context_prompt(
    tokenizer, example_labels: Sequence[str]
) -> tuple[list[int], list[tuple[int, int]]]:
    """Cell-typing prompt with labelled example cells (in-context learning).

    Same format as during training: each example is a line
    ``"Example cell k: The cell type is <label>."``, followed by the usual
    cell-typing question. The token span of each example line is conditioned by
    that example cell; every other token (system prompt, question, answer) by the
    query cell.

    Returns:
        Prompt token ids (ending at the assistant-turn opening) and, for each
        example, its ``(start, end)`` token span (end excluded).
    """
    user_content = ""
    char_spans = []
    for k, label in enumerate(example_labels):
        start = len(user_content)
        user_content += f"Example cell {k + 1}: {ANSWER_PREFIX}{label}."
        char_spans.append((start, len(user_content)))
        user_content += "\n"
    user_content += PROMPTS[TASK_CELL_TYPING]["user"]
    messages = [
        {"role": "system", "content": PROMPTS[TASK_CELL_TYPING]["system"]},
        {"role": "user", "content": user_content},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(
        prompt_text, add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = encoded["offset_mapping"]
    base = prompt_text.index(user_content)
    token_spans = []
    for char_start, char_end in char_spans:
        first, last = base + char_start, base + char_end
        tokens = [
            i for i, (a, b) in enumerate(offsets) if b > a and a < last and b > first
        ]
        token_spans.append((tokens[0], tokens[-1] + 1))
    return encoded["input_ids"], token_spans
