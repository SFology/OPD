# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from verl.utils.reward_score.ttrl_math import compute_score


@pytest.mark.parametrize(
    ("response", "ground_truth", "method"),
    [
        (r"Therefore, \\boxed{25}.", "025", "boxed"),
        ("Reasoning.\nAnswer: 25", "025", "explicit_answer"),
        ("Reasoning.\nFinal answer: $142$.", "142.0", "explicit_answer"),
        ("Reasoning.\nThe final answer is 204.", "204", "explicit_answer"),
        (r"Answer: \\frac{1}{2}", "1/2", "explicit_answer"),
    ],
)
def test_supported_final_answer_formats(response, ground_truth, method):
    result = compute_score(response, ground_truth, fast=True)
    assert result["acc"] is True
    assert result["format_score"] == 1.0
    assert result["extraction_method"] == method


def test_wrong_explicit_answer_is_parsed_but_incorrect():
    result = compute_score("Answer: 24", "025", fast=True)
    assert result["acc"] is False
    assert result["format_score"] == 1.0
    assert result["extraction_method"] == "explicit_answer"


@pytest.mark.parametrize(
    "response",
    [
        "The reasoning mentions 25 but never states a final answer.",
        "We tried 1, 2, and 3 before generation stopped",
        r"The final expression is \\boxed{25",
    ],
)
def test_reasoning_numbers_and_incomplete_boxes_are_not_guessed(response):
    result = compute_score(response, "025", fast=True)
    assert result["acc"] is False
    assert result["format_score"] == 0.0
    assert result["extraction_method"] is None
