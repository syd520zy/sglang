from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import torch

from sglang.srt.layers.attention.triton_backend import ForwardMetadata, TritonAttnBackend

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class TriAttentionTritonBackend(TritonAttnBackend):
    """A Triton decode backend that applies TriAttention-style sparse KV selection.

    This P0 implementation only changes normal decode path (no spec info, no sliding window).
    It keeps prefill/other modes identical to TritonAttnBackend.
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        # Selecting this backend means TriAttention is enabled.
        self.enable_triattention = True
        self.tri_window_size = model_runner.server_args.triattention_window_size
        self.tri_notable_budget = model_runner.server_args.triattention_notable_budget
        self.tri_selection_interval = model_runner.server_args.triattention_selection_interval

        self._warned_once = False

        if model_runner.use_mla_backend:
            logger.warning(
                "TriAttention backend currently supports non-MLA models only; falling back to Triton behavior."
            )
            self.enable_triattention = False

    def _can_apply_triattention(self, forward_batch: ForwardBatch) -> bool:
        if not self.enable_triattention:
            return False
        if not forward_batch.forward_mode.is_decode_or_idle():
            return False
        if forward_batch.spec_info is not None:
            return False
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            return False
        if self.tri_window_size <= 0 or self.tri_notable_budget <= 0:
            return False
        if self.tri_selection_interval <= 0:
            return False
        return True

    def _build_sparse_decode_indices(
        self, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build sparse decode kv indices: remote notable + local dense window."""
        bs = req_pool_indices.shape[0]
        req_ids: List[int] = req_pool_indices.tolist()
        seq_lens_cpu: List[int] = [int(x) for x in seq_lens.tolist()]

        selected_kv_chunks: List[torch.Tensor] = []
        selected_lens: List[int] = []

        for req_id, seq_len in zip(req_ids, seq_lens_cpu):
            if seq_len <= 0:
                selected_lens.append(0)
                continue

            local_start = max(0, seq_len - self.tri_window_size)
            # Approximate "update every interval steps" without per-request mutable state:
            # notable tokens are selected from a stable prefix whose end advances every
            # `tri_selection_interval` decode steps.
            stable_prefix_end = (
                local_start // self.tri_selection_interval
            ) * self.tri_selection_interval
            remote_pool_len = stable_prefix_end
            remote_count = min(self.tri_notable_budget, remote_pool_len)

            pieces: List[torch.Tensor] = []
            if remote_count > 0:
                if remote_count == remote_pool_len:
                    remote_pos = torch.arange(
                        remote_pool_len, device=self.device, dtype=torch.int64
                    )
                else:
                    remote_pos = torch.div(
                        torch.arange(
                            remote_count, device=self.device, dtype=torch.int64
                        )
                        * remote_pool_len,
                        remote_count,
                        rounding_mode="floor",
                    )
                pieces.append(remote_pos)

            local_pos = torch.arange(local_start, seq_len, device=self.device, dtype=torch.int64)
            if local_pos.numel() > 0:
                pieces.append(local_pos)

            if pieces:
                token_pos = torch.cat(pieces, dim=0)
                # Map logical token positions to physical KV pool locations.
                req_kv_indices = self.req_to_token[req_id, token_pos].to(torch.int64)
            else:
                req_kv_indices = torch.empty((0,), dtype=torch.int64, device=self.device)

            selected_kv_chunks.append(req_kv_indices)
            selected_lens.append(int(req_kv_indices.numel()))

        if selected_kv_chunks:
            kv_indices = torch.cat(selected_kv_chunks, dim=0)
        else:
            kv_indices = torch.empty((0,), dtype=torch.int64, device=self.device)

        selected_lens_t = torch.tensor(selected_lens, dtype=torch.int32, device=self.device)
        kv_indptr = self.kv_indptr
        kv_indptr[0] = 0
        if bs > 0:
            kv_indptr[1 : bs + 1] = torch.cumsum(selected_lens_t, dim=0)
        kv_indptr = kv_indptr[: bs + 1]

        return kv_indptr, kv_indices, selected_lens_t

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if not self._can_apply_triattention(forward_batch):
            return super().init_forward_metadata(forward_batch)

        bs = forward_batch.batch_size
        kv_indptr, kv_indices, selected_lens = self._build_sparse_decode_indices(
            forward_batch.req_pool_indices, forward_batch.seq_lens
        )

        attn_logits = torch.empty(
            (bs, self.num_head, self.max_kv_splits, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        attn_lse = torch.empty(
            (bs, self.num_head, self.max_kv_splits),
            dtype=torch.float32,
            device=self.device,
        )
        num_kv_splits = torch.empty((bs,), dtype=torch.int32, device=self.device)
        self.get_num_kv_splits(num_kv_splits, selected_lens)

        self.forward_metadata = ForwardMetadata(
            attn_logits=attn_logits,
            attn_lse=attn_lse,
            max_extend_len=None,
            num_kv_splits=num_kv_splits,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            qo_indptr=None,
            custom_mask=None,
            mask_indptr=None,
            window_kv_indptr=None,
            window_kv_indices=None,
            window_num_kv_splits=None,
            window_kv_offsets=None,
        )

        if not self._warned_once:
            logger.info(
                "TriAttention decode path enabled (window=%d, notable_budget=%d, interval=%d).",
                self.tri_window_size,
                self.tri_notable_budget,
                self.tri_selection_interval,
            )
            self._warned_once = True
