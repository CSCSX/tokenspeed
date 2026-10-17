# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Construct Host-only L3 storage from startup configuration and CPU facts."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from tokenspeed.runtime.cache.l3.backend import KvStoreStorage, MemoryKvStore

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from tokenspeed.runtime.cache.l3.config import L3AsyncConfig
    from tokenspeed.runtime.cache.l3.executor import L3Executor
    from tokenspeed.runtime.cache.transfer.host_storage import HostCacheBuild
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.utils.server_args import ServerArgs


def create_kvstore_storage_backend(
    backend_name: str | None,
    extra_config: str | None,
    *,
    host_buffer: Any,
    tp_size: int,
    pp_size: int,
) -> KvStoreStorage | None:
    """Return an L3 backend for ``backend_name``, or None when L3 is unset.

    ``memory`` is a test-only in-process store. ``mooncake`` is the production
    Mooncake Store client (SGLang/vLLM HiCache equivalent). ``tp_size`` and
    ``pp_size`` divide ``global_segment_size`` so every cache-owning rank
    (attention TP × PP) mounts an equal share of the configured total. Pass
    the resolved ``mapping.attn.tp_size``.
    """

    if backend_name is None:
        return None
    name = backend_name.strip().lower()
    if name == "memory":
        return MemoryKvStore()
    if name == "mooncake":
        from tokenspeed.runtime.cache.l3.mooncake import (
            MooncakeKvStore,
            parse_extra_config,
        )

        return MooncakeKvStore(
            extra_config=parse_extra_config(extra_config),
            host_buffer=host_buffer,
            tp_size=tp_size,
            pp_size=pp_size,
        )
    raise ValueError(f"unsupported KVStore storage backend {backend_name!r}")


def build_l3_cache(
    server_args: ServerArgs,
    model_config: ModelConfig,
    draft_model_config: ModelConfig | None,
    *,
    host_cache: HostCacheBuild | None,
    world_cpu_group: ProcessGroup | None,
) -> tuple[L3AsyncConfig | None, L3Executor | None]:
    """Build an independent L3 executor without retaining device capabilities.

    Args:
        server_args: Resolved launch settings; numerical choices are snapshotted.
        model_config: Target checkpoint configuration, without model weights.
        draft_model_config: Draft checkpoint configuration, or None.
        host_cache: Startup-only Host resource and resolved byte/producer facts.
            None when device construction produced no Host L2 allocation.
        world_cpu_group: The existing Gloo world group for checkpoint agreement;
            None in a single-process deployment.

    Returns:
        Async limits and their Host-only executor. No L2 executor, device handle
        or attention buffer is retained. Namespace inputs are frozen before
        the backend is created and workers start. Both values are None when
        no storage backend is configured; no client, worker or collective runs.
    """
    if server_args.kvstore_storage_backend is None:
        return None, None
    if host_cache is None:
        raise ValueError("L3 construction requires a Host L2 allocation")
    host_storage = host_cache.storage
    ownership = host_cache.ownership
    cache_fields = tuple(
        tuple(field.field_id for field in group.fields) for group in host_cache.groups
    )
    transfer_schema = host_cache.transfer_schema
    if len(cache_fields) != len(ownership.shard_counts):
        raise ValueError("L3 field groups and Host ownership must agree")
    for fields, shards in zip(cache_fields, ownership.shard_counts, strict=True):
        if shards > 1:
            for field_id in fields:
                if transfer_schema.partition_for(field_id) is not None:
                    raise ValueError(
                        "L3 page sharding requires replicated field payloads; "
                        f"{field_id} is head-partitioned"
                    )

    eagle3_layers_to_capture = host_cache.eagle3_layers_to_capture
    attention_backend_name = host_cache.attention_backend_name
    draft_attention_backend_name = host_cache.draft_attention_backend_name
    from tokenspeed.runtime.cache.l3.backend import cache_layout_signature
    from tokenspeed.runtime.cache.l3.config import L3AsyncConfig

    cache_signature = cache_layout_signature(
        host_cache,
        cache_dtype=f"{server_args.kv_cache_dtype}:{model_config.dtype}",
    )
    l3_config = L3AsyncConfig.from_extra_config(
        server_args.kvstore_storage_backend_extra_config,
        host_blocks=host_storage.num_host_lcm_blocks,
        prefix_granularity=host_cache.prefix_granularity,
    )
    import torch.distributed as dist

    from tokenspeed.runtime.cache.l3.backend import (
        L3_RUNTIME_COMPAT,
        l3_cache_quantization_id,
        l3_checkpoint_id,
        l3_kv_producer_id,
        share_l3_checkpoint_ids,
        storage_key_prefix,
    )

    initialized = dist.is_available() and dist.is_initialized()
    if world_cpu_group is not None:
        if not initialized or dist.get_backend(world_cpu_group) != "gloo":
            raise ValueError("L3 checkpoint agreement requires a Gloo CPU group")
        world_size = dist.get_world_size(world_cpu_group)
        rank = dist.get_rank(world_cpu_group)
    else:
        if initialized and dist.get_world_size() > 1:
            raise ValueError("L3 checkpoint agreement requires an explicit CPU group")
        world_size, rank = 1, 0
    checkpoint_id = l3_checkpoint_id(
        model_config.model_path,
        hf_config=model_config.hf_config,
        revision=str(model_config.revision or ""),
        load_format=str(server_args.load_format),
        # LoadConfig currently gets the same empty extra-config;
        # both must stay aligned if a shard pattern is wired through.
        model_loader_extra_config={},
        ext_yaml=str(server_args.ext_yaml or ""),
    )
    if draft_model_config is not None:
        draft_revision = l3_checkpoint_id(
            draft_model_config.model_path,
            hf_config=draft_model_config.hf_config,
            revision=str(draft_model_config.revision or ""),
            load_format=str(server_args.load_format),
            model_loader_extra_config={},
            ext_yaml=str(server_args.ext_yaml or ""),
        )
    else:
        draft_revision = ""

    def gather_checkpoint_ids(payload: list) -> list:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, payload, group=world_cpu_group)
        return gathered

    checkpoint_id, draft_revision = share_l3_checkpoint_ids(
        [checkpoint_id, draft_revision],
        rank=rank,
        world_size=world_size,
        gather=gather_checkpoint_ids,
    )
    pipeline_rank = server_args.mapping.pp_rank if server_args.mapping.has_pp else 0
    if draft_model_config is not None:
        draft_model = str(draft_model_config.model_path)
        draft_quantization = str(draft_model_config.quantization or "")
    else:
        draft_model = ""
        draft_quantization = ""
    cache_quantization = l3_cache_quantization_id(
        quantization=str(model_config.quantization or ""),
        quantization_param_path=str(server_args.quantization_param_path or ""),
        draft_quantization=draft_quantization,
    )
    attn_tp_size = int(server_args.mapping.attn.tp_size)
    kv_producer = l3_kv_producer_id(
        server_args,
        dtype=str(model_config.dtype),
        draft_dtype=(
            str(draft_model_config.dtype) if draft_model_config is not None else ""
        ),
    )

    model_name = str(server_args.model)
    model_overrides = copy.deepcopy(model_config.model_override_args)
    skip_softmax_threshold = float(server_args.skip_softmax_threshold)

    def prefix_for_weight_version(weight_version: str) -> str:
        return storage_key_prefix(
            model_name,
            revision=checkpoint_id,
            weight_version=weight_version,
            model_overrides=model_overrides,
            cache_signature=cache_signature,
            pipeline_rank=pipeline_rank,
            attn_tp_size=attn_tp_size,
            draft_model=draft_model,
            draft_revision=draft_revision,
            draft_weight_version=weight_version if draft_model else "",
            cache_quantization=cache_quantization,
            runtime_compat=L3_RUNTIME_COMPAT,
            attention_backend=attention_backend_name,
            draft_attention_backend=draft_attention_backend_name,
            skip_softmax_threshold=skip_softmax_threshold,
            eagle3_layers_to_capture=eagle3_layers_to_capture,
            kv_producer=kv_producer,
        )

    from tokenspeed.runtime.cache.l3.executor import L3Executor
    from tokenspeed.runtime.cache.l3.host_store import L3HostStore

    key_prefix = prefix_for_weight_version(server_args.weight_version)
    storage_backend = create_kvstore_storage_backend(
        server_args.kvstore_storage_backend,
        server_args.kvstore_storage_backend_extra_config,
        host_buffer=host_storage.host_buffer,
        tp_size=server_args.mapping.attn.tp_size,
        pp_size=(server_args.mapping.pp_size if server_args.mapping.has_pp else 1),
    )
    l3_store = L3HostStore(
        storage_backend,
        host_storage,
        key_prefix=key_prefix,
        rank=server_args.mapping.attn.tp_rank,
        ownership=ownership,
    )
    l3_executor = L3Executor(
        l3_store,
        config=l3_config,
        prefix_for_weight_version=prefix_for_weight_version,
    )

    return l3_config, l3_executor
