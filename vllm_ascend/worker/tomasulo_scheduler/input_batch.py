# SPDX-License-Identifier: Apache-2.0
"""Single-forward input payload for the Tomasulo scheduler.

Assembler builds a ``TomasuloInputBatch`` per forward step and hands it to
Executor, which unpacks the fields into ``set_ascend_forward_context`` and
``NPUModelRunner._model_forward``.
"""

from dataclasses import dataclass

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.attention.backend import AttentionMetadata


@dataclass
class TomasuloInputBatch:
    input_ids: torch.Tensor
    positions: torch.Tensor
    attn_metadata: AttentionMetadata
    num_tokens_padded: int
    num_tokens_across_dp: torch.Tensor | None
    cudagraph_mode: CUDAGraphMode
    batch_desc: BatchDescriptor
    num_actual_tokens: int
