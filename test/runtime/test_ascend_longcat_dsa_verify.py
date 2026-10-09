# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT

"""The eager DCP verify fallback must expose only each candidate prefix."""

from types import SimpleNamespace

import torch

from tokenspeed.runtime.layers.attention.backends.paged import ascend_longcat_dsa


def test_dcp_verify_advances_visible_lengths_and_restores_metadata(monkeypatch):
    monkeypatch.setattr(ascend_longcat_dsa, "get_is_capture_mode", lambda: False)
    width, batch = 4, 2
    query = torch.arange(batch * width * 2, dtype=torch.float32).view(
        batch * width, 1, 2
    )
    locations = torch.arange(batch * width, dtype=torch.int32)
    metadata = SimpleNamespace(
        seq_lens=torch.tensor([8, 9], dtype=torch.int32),
        page_table=torch.ones((batch, 1), dtype=torch.int32),
    )

    class FakeDcp:
        degree = 8

        def __init__(self):
            self.lengths = []

        def refresh_metadata(self, *, seq_lens, **kwargs):
            del kwargs
            self.lengths.append(seq_lens.tolist())

    class FakeBackend:
        _dcp = FakeDcp()
        kernel_page_size = 128
        index_init_tokens = 1
        index_local_tokens = 1

        def _select_indexed(
            self,
            q,
            k,
            layer,
            out_cache_loc,
            pool,
            q_ends,
            kv_lengths,
            table,
            kwargs,
            **options,
        ):
            del k, layer, pool, table
            assert options["context_parallel"]
            assert q_ends.tolist() == [1, 2]
            assert out_cache_loc.tolist() == kwargs["index_query"][:, 0].tolist()
            assert kwargs["index_weights"].shape[0] == batch
            assert kv_lengths.tolist() == self._dcp.lengths[-1]
            return q

        def _run_indexed_sparse_attention(
            self, selection, layer, pool, *, head_major_output
        ):
            del layer, pool, head_major_output
            return selection.flatten(1)

    backend = FakeBackend()
    actual = ascend_longcat_dsa.AscendDSABackend._forward_verify_dcp(
        backend,
        query,
        query,
        None,
        locations,
        None,
        batch,
        metadata,
        0,
        width,
        False,
        True,
        {
            "index_query": locations[:, None],
            "index_key": query,
            "index_weights": query,
        },
    )
    assert backend._dcp.lengths == [[5, 6], [6, 7], [7, 8], [8, 9], [8, 9]]
    torch.testing.assert_close(actual, query.flatten(1))
