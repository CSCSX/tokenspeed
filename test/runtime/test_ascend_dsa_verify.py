# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT

"""Ascend DSA gives each verify candidate its own causal DCP metadata."""

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.registry import KernelRegistry

from tokenspeed.runtime.layers.attention import backends
from tokenspeed.runtime.layers.attention.backends.paged import ascend_dsa
from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
from tokenspeed.runtime.layers.attention.dcp.metadata import (
    PositionPreservingDCPMetadata,
)


def test_ascend_backend_reuses_the_common_dsa_base():
    assert issubclass(ascend_dsa.AscendDSABackend, DSABackend)


def test_both_dsa_modules_are_imported():
    assert backends.ascend_dsa is ascend_dsa
    assert backends.dsa.DSABackend is DSABackend


def test_ascend_mla_prologue_is_limited_to_native_absorbed_kv():
    spec = KernelRegistry.get().get_by_name("ascend_composite_mla_prologue")
    assert spec is not None
    assert spec.capability.vendors == frozenset({"ascend"})
    assert spec.traits["expanded"] == frozenset({False})
    assert spec.traits["kv_format"] == frozenset({"native"})
    assert spec.traits["store"] == frozenset({True})


def test_indexer_writes_only_its_own_cache_after_mla_prologue():
    writes = []
    index_cache = torch.empty((1, 128, 1, 32))

    class Kernels:
        def scatter(self, rows, cache, locations):
            writes.append((rows, cache, locations))

        def index(self, *args):
            return torch.zeros((1, 1), dtype=torch.int32), torch.ones(
                (1,), dtype=torch.int32
            )

    class Pool:
        def get_key_buffer(self, layer_id):
            raise AssertionError("MLA prologue already wrote the KV cache")

        def get_component(self, layer_id, name):
            assert name == "dsa_index_k"
            return index_cache

    backend = SimpleNamespace(
        _indexer_spec=SimpleNamespace(index_head_dim=32, index_topk=1),
        _indexer_kernels=Kernels(),
        kernel_page_size=128,
        step_counter=None,
        _dcp=SimpleNamespace(degree=1),
        index_init_tokens=0,
        index_local_tokens=0,
    )
    key = torch.ones((1, 1, 32))
    locations = torch.tensor([3], dtype=torch.int32)
    ascend_dsa.AscendDSABackend._select_indexed(
        backend,
        torch.ones((1, 1, 8)),
        None,
        SimpleNamespace(layer_id=0),
        locations,
        Pool(),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([[0]], dtype=torch.int32),
        {
            "index_key": key,
            "index_query": torch.ones((1, 1, 32)),
            "index_weights": torch.ones((1, 1)),
        },
    )
    assert len(writes) == 1
    assert writes[0][1].data_ptr() == index_cache.data_ptr()
    assert writes[0][2] is locations


def test_each_candidate_keeps_its_own_local_page_prefix():
    table = torch.tensor([[1, 2, 3]] * 4, dtype=torch.int32)
    placement = PositionPreservingDCPMetadata(
        virtual_page_table=table,
        local_page_table=table,
        virtual_block_count=8,
        degree=8,
        rank=1,
        owner_mask=torch.tensor([[False, True, False]] * 4),
    )
    compact = ascend_dsa._compact_dcp_kernel_metadata(
        placement=placement,
        seq_lens=torch.tensor([254, 255, 256, 257], dtype=torch.int32),
        page_size=128,
        init_tokens=0,
        local_tokens=0,
    )
    assert compact.seq_lens.tolist() == [126, 127, 128, 128]
    assert compact.page_table[:, 0].tolist() == [2, 2, 2, 2]


def test_decode_metadata_storage_is_stable_across_batch_sizes():
    dcp = ascend_dsa._AscendDSAContextParallel(
        degree=1,
        rank=0,
        ranks=(0,),
        auxiliary_namespace="test_dsa_cp",
        virtual_block_count=8,
    )
    dcp.allocate_decode_buffers(rows=8, columns=3, device="cpu")
    large = dcp._copy_buffer("query_page_table", torch.ones((8, 3), dtype=torch.int32))
    small = dcp._copy_buffer("query_page_table", torch.zeros((2, 3), dtype=torch.int32))
    assert large.data_ptr() == small.data_ptr()
    assert small.tolist() == [[0, 0, 0], [0, 0, 0]]
    with pytest.raises(RuntimeError, match="preallocated decode geometry"):
        dcp._copy_buffer("query_page_table", torch.ones((9, 3), dtype=torch.int32))


@pytest.mark.parametrize("width", [2, 4, 8])
def test_dcp_verify_batches_query_rows_with_causal_lengths(width):
    lengths = torch.tensor([20, 129, 260], dtype=torch.int32)
    table = torch.arange(9, dtype=torch.int32).view(3, 3)
    metadata = SimpleNamespace(seq_lens=lengths, page_table=table, num_extends=1)

    class FakeDcp:
        degree = 8

        def __init__(self):
            self.refreshed = []

        def _copy_buffer(self, name, value):
            del name
            return value.clone()

        def refresh_metadata(self, *, seq_lens, page_table, **kwargs):
            del kwargs
            self.refreshed.append((seq_lens.clone(), page_table.clone()))

    class FakeBackend:
        spec_num_tokens = width
        kernel_page_size = 128
        index_init_tokens = 1
        index_local_tokens = 1
        forward_decode_metadata = metadata

        def __init__(self):
            self._dcp = FakeDcp()
            self._dense_backend = SimpleNamespace(
                refresh_decode_metadata=lambda *args, **kwargs: None
            )
            self._decode_query_seq_lens = None
            self._decode_query_page_table = None
            self.selections = []

        def _validate_logit_cap(self, value):
            assert value == 0

        def _select_indexed(
            self,
            q,
            k,
            layer,
            locations,
            pool,
            q_ends,
            kv_lengths,
            pages,
            kwargs,
            **options,
        ):
            del k, layer, pool, kwargs
            self.selections.append(
                (
                    q_ends.clone(),
                    kv_lengths.clone(),
                    pages.clone(),
                    locations.clone(),
                    options,
                )
            )
            return q

        def _run_indexed_sparse_attention(
            self, selection, layer, pool, *, head_major_output
        ):
            del layer, pool, head_major_output
            return selection.flatten(1)

    backend = FakeBackend()
    ascend_dsa.AscendDSABackend.refresh_decode_metadata(backend, 3, 3, lengths, table)
    expected_lengths = torch.cat(
        [torch.arange(int(end) - width + 1, int(end) + 1) for end in lengths]
    ).to(torch.int32)
    torch.testing.assert_close(backend._decode_query_seq_lens, expected_lengths)
    torch.testing.assert_close(
        backend._decode_query_page_table, table.repeat_interleave(width, dim=0)
    )
    torch.testing.assert_close(backend._dcp.refreshed[0][0], expected_lengths)

    rows = 2 * width
    query = torch.arange(rows * 2, dtype=torch.float32).view(rows, 1, 2)
    locations = torch.arange(rows, dtype=torch.int32)
    output = ascend_dsa.AscendDSABackend.forward_decode(
        backend,
        query,
        query,
        None,
        SimpleNamespace(logit_cap=0),
        locations,
        None,
        2,
    )
    torch.testing.assert_close(output, query.flatten(1))
    assert len(backend.selections) == 1
    q_ends, visible, pages, selected_locations, options = backend.selections[0]
    assert q_ends.tolist() == list(range(1, rows + 1))
    torch.testing.assert_close(visible, expected_lengths[width:])
    torch.testing.assert_close(pages, table[1:].repeat_interleave(width, dim=0))
    torch.testing.assert_close(selected_locations, locations)
    assert options["context_parallel_row_start"] == width


def test_single_token_mixed_decode_restores_request_shaped_dcp_metadata():
    metadata = SimpleNamespace(
        seq_lens=torch.tensor([12, 24], dtype=torch.int32),
        page_table=torch.tensor([[1], [2]], dtype=torch.int32),
        num_extends=1,
    )

    class FakeDcp:
        degree = 8

        def __init__(self):
            self.refreshed = []

        def refresh_metadata(self, *, seq_lens, page_table, **kwargs):
            del kwargs
            self.refreshed.append((seq_lens.clone(), page_table.clone()))

    class FakeBackend:
        spec_num_tokens = 4
        kernel_page_size = 128
        index_init_tokens = 1
        index_local_tokens = 1
        forward_decode_metadata = metadata

        def __init__(self):
            self._dcp = FakeDcp()

        def _validate_logit_cap(self, value):
            assert value == 0

        def _select_indexed(
            self,
            q,
            k,
            layer,
            locations,
            pool,
            q_ends,
            kv_lengths,
            pages,
            kwargs,
            **options,
        ):
            del k, layer, locations, pool, kwargs
            assert q_ends.tolist() == [1]
            assert kv_lengths.tolist() == [24]
            assert pages.tolist() == [[2]]
            assert options["context_parallel_row_start"] == 1
            return q

        def _run_indexed_sparse_attention(
            self, selection, layer, pool, *, head_major_output
        ):
            del layer, pool, head_major_output
            return selection.flatten(1)

    backend = FakeBackend()
    query = torch.ones((1, 1, 2))
    ascend_dsa.AscendDSABackend.forward_decode(
        backend,
        query,
        query,
        None,
        SimpleNamespace(logit_cap=0),
        torch.tensor([7], dtype=torch.int32),
        None,
        1,
    )
    assert backend._dcp.refreshed[0][0].tolist() == [12, 24]
