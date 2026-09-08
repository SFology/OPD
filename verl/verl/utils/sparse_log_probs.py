# Copyright 2026 Individual Contributor: SFology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Exact log-probability gathers for sparse trajectory action requests."""

from __future__ import annotations

import torch


def gather_sparse_response_log_probs(
    logits: torch.Tensor,
    positions: torch.Tensor,
    action_ids: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Score packed ``(response position, action)`` requests from padded logits.

    ``logits`` has shape ``[batch, response_length, vocabulary]`` while each
    request tensor has shape ``[batch, max_requests_per_trajectory]``. The
    normalizer is evaluated once per unique requested state, even when many
    anchor actions point to that state.
    """

    output = torch.zeros_like(positions, dtype=torch.float32)
    valid_rows, valid_slots = valid.nonzero(as_tuple=True)
    if valid_rows.numel() == 0:
        return output

    response_length = logits.shape[1]
    requested_positions = positions[valid_rows, valid_slots]
    requested_actions = action_ids[valid_rows, valid_slots]
    flat_states = valid_rows * response_length + requested_positions
    unique_states, inverse = torch.unique(flat_states, sorted=True, return_inverse=True)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    normalizers = torch.logsumexp(flat_logits[unique_states], dim=-1)
    values = flat_logits[flat_states, requested_actions].float() - normalizers[inverse].float()
    output[valid_rows, valid_slots] = values
    return output


def gather_sparse_packed_response_log_probs(
    logits: torch.Tensor,
    unpadded_indices: torch.Tensor,
    *,
    sequence_length: int,
    response_length: int,
    positions: torch.Tensor,
    action_ids: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Score sparse requests from remove-padding logits.

    ``unpadded_indices`` maps packed rows back to the flattened padded input.
    The logit immediately before response token ``t`` is the predictor for the
    action at ``t``.
    """

    output = torch.zeros_like(positions, dtype=torch.float32)
    valid_rows, valid_slots = valid.nonzero(as_tuple=True)
    if valid_rows.numel() == 0:
        return output

    full_positions = (
        valid_rows * sequence_length
        + (sequence_length - response_length - 1)
        + positions[valid_rows, valid_slots]
    )
    packed_positions = torch.searchsorted(unpadded_indices, full_positions)
    requested_actions = action_ids[valid_rows, valid_slots]
    unique_positions, inverse = torch.unique(packed_positions, sorted=True, return_inverse=True)
    normalizers = torch.logsumexp(logits[unique_positions], dim=-1)
    values = logits[packed_positions, requested_actions].float() - normalizers[inverse].float()
    output[valid_rows, valid_slots] = values
    return output
