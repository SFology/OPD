# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except Exception in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides a math answer grading function with high recall.
Based on HF math_verify, verl, open reasoner zero, etc.
"""

import re
import traceback

from latex2sympy2_extended import latex2sympy
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr

from .math_utils import extract_boxed_answer, grade_answer_mathd, grade_answer_sympy, is_latex_equal, timeout_ours

"""
This code is adapted from Entropy Machanism Recipe (https://github.com/volcengine/verl/tree/main/recipe/entropy/).
"""

_EXPLICIT_ANSWER_PATTERNS = (
    re.compile(r"(?im)^\s*(?:final\s+answer|answer)\s*:\s*(.+?)\s*$"),
    re.compile(r"(?im)^\s*(?:the\s+)?final\s+answer\s+is\s+(.+?)\s*$"),
)


def _trim_explicit_answer(answer: str) -> str:
    """Remove presentation-only delimiters without guessing from reasoning text."""
    answer = answer.strip()
    if answer.startswith("$") and answer.endswith("$") and len(answer) >= 2:
        answer = answer[1:-1].strip()
    return answer.rstrip().rstrip(".。")


def extract_answer_with_method(passage: str) -> tuple[str | None, str | None]:
    """Extract a boxed or explicitly labelled final answer.

    We intentionally do not fall back to the last number in a response: doing so
    silently turns intermediate reasoning into a prediction when generation is
    incomplete.
    """
    if "\\boxed" in passage or "\\fbox" in passage:
        boxed = extract_boxed_answer(passage)
        if boxed is not None:
            return boxed, "boxed"
    for pattern in _EXPLICIT_ANSWER_PATTERNS:
        matches = pattern.findall(passage)
        if matches:
            answer = _trim_explicit_answer(matches[-1])
            if answer:
                return answer, "explicit_answer"
    return None, None


def extract_answer(passage: str) -> str | None:
    answer, _ = extract_answer_with_method(passage)
    return answer


def grade(model_answer: str, gt_answer: str, fast: bool = True):
    if "\\boxed" in gt_answer:
        gt_answer = extract_answer(gt_answer)
    correct = grade_answer_mathd(model_answer, gt_answer) or grade_answer_sympy(model_answer, gt_answer)
    if not fast:
        # This mode further uses math_verify to recall originally false positives.
        # Will be a bit slower, and sensitive to bad inputs.
        correct = correct or is_latex_equal(
            model_answer,
            gt_answer,
        )
    return correct

@timeout_ours(timeout_seconds=10)
def simplify_expression_string(expression_string: str) -> str:
    try:
        sympy_expr = parse_expr(expression_string, transformations="all", evaluate=False)
        simplified_expr = simplify(sympy_expr)
        return str(simplified_expr)
    except TimeoutError:
        return expression_string
    except Exception:
        try:
            sympy_expr = latex2sympy(expression_string)
            simplified_expr = simplify(sympy_expr)
            return str(simplified_expr)
        except TimeoutError:
            return expression_string
        except Exception:
            return expression_string

def compute_score(model_response, gt_answer, fast=False):
    model_answer, extraction_method = extract_answer_with_method(model_response)

    if model_answer is None:
        return {
            "score": 0.0,
            "format_score": 0.0,
            "acc": False,
            "extracted_gt": gt_answer,
            "pred": "",
            "extraction_method": None,
        }
        # return 0.0, 0.0  # Cannot even parse anything.
    is_correct = False
    if isinstance(gt_answer, float) or isinstance(gt_answer, int):
        gt_answer = str(gt_answer)
    if isinstance(gt_answer, str):
        is_correct = grade(model_answer, gt_answer, fast)
    elif isinstance(gt_answer, list):
        is_correct = False
        for gt in gt_answer:
            is_correct |= grade(model_answer, gt, fast)
    if is_correct:
        return {
            "score": 1.0,
            "format_score": 1.0,
            "acc": True,
            "extracted_gt": gt_answer,
            "pred": model_answer,
            "extraction_method": extraction_method,
        }
    else:
        return {
            "score": 0.0,
            "format_score": 1.0,
            "acc": False,
            "extracted_gt": gt_answer,
            "pred": model_answer,
            "extraction_method": extraction_method,
        }

def reward_func(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    try:
        res = compute_score(solution_str, str(ground_truth))

        if isinstance(res, dict):
            return res
        elif isinstance(res, (int, float, bool)):
            return float(res)
        else:
            return float(res[0])
    except Exception as e:
        print(f"[ERROR] Error in process_completion for task : {str(e)}")
        traceback.print_exc()
        raise
