# Copyright 2026 InstaDeep Ltd
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
"""Turn generated answers into labels (same parsing as the CellScribe evaluation)."""

import re
from typing import Optional


def extract_predicted_cell_type(
    generated_text: str, possible_cell_types: Optional[list[str]] = None
) -> str:
    """
    Parses the generated text to find the predicted cell type.

    With ``possible_cell_types`` (e.g. the training vocabulary, as in the paper's
    evaluation), returns the longest label contained in the answer
    (case-insensitive), or "Unknown". Without it, returns the text following
    "The cell type is".
    """
    try:
        text_cleaned = generated_text
        for special_token in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
            text_cleaned = text_cleaned.replace(special_token, "")

        if "assistant" in text_cleaned:
            answer_part_original = text_cleaned.split("assistant")[-1].strip()
        else:
            answer_part_original = text_cleaned.strip()
        answer_part_lower = answer_part_original.lower()

        if possible_cell_types:
            # Longest first, to match the most specific label.
            sorted_cell_types = sorted(possible_cell_types, key=len, reverse=True)
            for cell_type in sorted_cell_types:
                if cell_type.lower() in answer_part_lower:
                    return cell_type
            return "Unknown"

        patterns = [
            "the cell type is a ",
            "the cell type is ",
            " cell type is a ",
            " cell type is ",
            "cell type is a ",
            "cell type is ",
        ]
        for pattern in patterns:
            if pattern in answer_part_lower:
                pos = answer_part_lower.find(pattern) + len(pattern)
                extracted = answer_part_original[pos:]
                cleaned = extracted.strip().rstrip(".!")
                cleaned = cleaned.split("\n")[0].split("  ")[0]
                if cleaned:
                    return cleaned

        return "Unknown"

    except Exception:  # noqa
        return "Unknown"


# A ``program (score)`` pair. The name may contain spaces, ``/``, ``-``, ``+``,
# ``,``, ``.`` and parentheses (e.g. ``Cell cycle (HMG-rich)``); it excludes ``:``
# so that the ``Active programs:`` prefix is not absorbed into a name.
_PAIR_RE = re.compile(r"([A-Za-z0-9][A-Za-z0-9 +/()'\-,.]*?)\s*\((-?\d+\.?\d*)\)")
_NAME_PREFIX_RE = re.compile(r"^(?:active programs?|highest|and)\s+", re.IGNORECASE)


def _clean_program_name(name: str) -> str:
    prev = None
    while prev != name:
        prev = name
        name = _NAME_PREFIX_RE.sub("", name).strip(" ,.")
    return name


def parse_cell_state_prediction(
    text: str,
) -> tuple[list[tuple[str, float]], Optional[str]]:
    """Parse a cell-state answer into ``(program, score)`` pairs + hallmark::

        Active programs: Cell Cycle - G2/M (1.9), EMT-I (1.1).
        Dominant hallmark: Cell Cycle.

        No strongly active programs detected. Highest: Cell Cycle - G2/M (0.3).
        Dominant hallmark: Cell Cycle.

    Programs are returned in the generated order (dominant first).
    """
    programs: list[tuple[str, float]] = []
    dominant_hallmark: Optional[str] = None

    hallmark_match = re.search(r"Dominant hallmark:\s*(.+?)\.", text)
    if hallmark_match:
        dominant_hallmark = hallmark_match.group(1).strip()
    core = re.split(r"Dominant hallmark:", text, maxsplit=1)[0]

    for m in _PAIR_RE.finditer(core):
        name = _clean_program_name(m.group(1))
        if name:
            programs.append((name, float(m.group(2))))
    return programs, dominant_hallmark
