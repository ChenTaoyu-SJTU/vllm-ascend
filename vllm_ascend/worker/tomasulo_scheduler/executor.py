# SPDX-License-Identifier: Apache-2.0
"""Minimal forward/logits/sample executor for the Tomasulo scheduler.

The executor is intentionally stateless: each call is fully described by its
arguments (``TomasuloInputBatch`` / ``hidden_states`` / ``logits_indices`` /
``sampling_metadata``). It exposes three atomic primitives that correspond to
the three stages of a single decode step:

* ``forward``        -- run ``NPUModelRunner._model_forward`` inside an
                        ``set_ascend_forward_context`` block, after refreshing
                        the RoPE cos/sin cache.
* ``compute_logits`` -- gather the per-request sampling positions from the
                        hidden states and project them to vocab space via
                        ``self._model.compute_logits``.
* ``sample``         -- run the ``AscendSampler`` on the logits and return the
                        sampled token ids.

The executor intentionally does NOT wrap ``NPUModelRunner.sample_tokens``; that
method carries PP / grammar / spec-decode / bookkeeping / logprobs / EPLB /
async-output logic that the demo has explicitly excluded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.worker.tomasulo_scheduler.input_batch import TomasuloInputBatch

if TYPE_CHECKING:
    from vllm.v1.sample.metadata import SamplingMetadata

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TomasuloExecutor:
    """Stateless wrapper around forward + compute_logits + sample.

    Parameters
    ----------
    model_runner:
        An already-initialised ``NPUModelRunner``. The runner MUST have
        completed ``load_model()`` so that ``model_runner.model`` and
        ``model_runner.sampler`` are available.
    """

    def __init__(self, model_runner: "NPUModelRunner") -> None:
        self.model_runner = model_runner
        self._vllm_config = model_runner.vllm_config
        self._model = model_runner.model
        self._model_forward = model_runner._model_forward
        self._sampler = model_runner.sampler

    def forward(self, input_batch: TomasuloInputBatch) -> torch.Tensor:
        """Run one ``_model_forward`` pass and return the raw hidden states.

        The RoPE cos/sin cache is refreshed before entering the forward
        context so that attention layers observe up-to-date rotary tables;
        this mirrors the main ``execute_model`` path.
        """
        update_cos_sin(input_batch.positions)

        with set_ascend_forward_context(
            input_batch.attn_metadata,
            self._vllm_config,
            num_tokens=input_batch.num_tokens_padded,
            num_tokens_across_dp=input_batch.num_tokens_across_dp,
            aclgraph_runtime_mode=input_batch.cudagraph_mode,
            batch_descriptor=input_batch.batch_desc,
            num_actual_tokens=input_batch.num_actual_tokens,
            model_instance=self._model,
            max_tokens_across_pcp=0,
            skip_compiled=False,
        ):
            hidden_states = self._model_forward(
                input_batch.num_tokens_padded,
                input_batch.input_ids,
                input_batch.positions,
                intermediate_tensors=None,
                inputs_embeds=None,
            )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Gather sampling positions and project to vocab space.

        ``logits_indices`` is a 1-D int64 tensor of shape ``[num_reqs]`` whose
        entries are offsets into ``hidden_states``; the typical source is
        ``query_start_loc.gpu[1 : num_reqs + 1] - 1`` (the last token of each
        sequence in the batch).
        """
        sample_hidden_states = hidden_states[logits_indices]
        logits = self._model.compute_logits(sample_hidden_states)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: "SamplingMetadata",
    ) -> torch.Tensor:
        """Sample one token per request and return only ``sampled_token_ids``.

        The returned tensor has shape ``[num_reqs, 1]`` and dtype ``int32``
        (see ``vllm.v1.sample.sampler.Sampler.forward``). Logprobs, grammar
        bitmasks, speculative decoding and async-output wrapping are
        deliberately skipped.
        """
        sampler_output = self._sampler(logits, sampling_metadata)
        return sampler_output.sampled_token_ids
