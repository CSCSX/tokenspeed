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

"""Resolve logical cache objects to immutable keys and local Host byte ranges."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from tokenspeed.runtime.cache.l3.backend import KvStoreStorage, storage_object_key

if TYPE_CHECKING:
    from tokenspeed.runtime.cache.transfer.ownership import BlockOwnerTranslation

logger = logging.getLogger(__name__)

StoragePage = tuple[
    int, int, str, int
]  # group_index, host_block_id, content_hash, page_offset


class L3HostStore:
    """Zero-copy adapter from compact Host pages to a ``KvStoreStorage``."""

    def __init__(
        self,
        backend: KvStoreStorage,
        host_storage: Any,
        *,
        key_prefix: str,
        rank: int,
        ownership: BlockOwnerTranslation | None,
    ):
        self.backend = backend
        self.host_storage = host_storage
        self._base_key_prefix = key_prefix
        self.rank = int(rank)
        self.ownership = ownership

    @property
    def key_prefix(self) -> str:
        return self._base_key_prefix

    def set_key_prefix(self, key_prefix: str) -> None:
        """Publish and restore under a new namespace without deleting it.

        Live weight updates change ``weight_version`` after Device/Host
        have been flushed and the GPU load succeeds. Flush /
        ``rotate_namespace`` still deletes the *current* prefix first so
        stale KV is gone before this switches the writers.
        """

        if not key_prefix:
            raise ValueError("key_prefix must be non-empty")
        self._base_key_prefix = key_prefix

    def rotate_namespace(self) -> bool:
        """Delete objects under this stable namespace.

        The prefix does not change, so a restarted rank in *this* job still
        probes the same keys and observes the deletion through
        ``batch_exists``. TP/CP/PP/DP MIN-reduce clearability and the
        delete result inside one TokenSpeed process group. Independent
        deployments that share a tenant are not in those groups: a
        fleet-wide wipe is an operator flush of every instance. A peer
        PUT after this delete is a later ``batch_exists`` hit, not a
        lease; vanished-L3 prefetch recovers a miss.

        Returns True when the store reports the prefix is gone. A False
        must not be followed by Device/Host ``ClearCache``.
        """

        try:
            return bool(self.backend.remove_by_prefix(f"{self.key_prefix}_"))
        except Exception:
            logger.exception("L3 namespace delete failed")
            return False

    def object_key(self, content_hash: str, group_id: int, page_offset: int) -> str:
        return storage_object_key(
            content_hash,
            group_id,
            page_offset,
            prefix=self.key_prefix,
            rank=(
                self.rank // self.ownership.shard_counts[group_id]
                if self.ownership is not None
                else self.rank
            ),
        )

    def page_ranges(
        self, pages: Sequence[StoragePage]
    ) -> tuple[list[str], list[int | None], list[int]]:
        """Capture local Host ranges in logical-key order.

        A None offset means this rank owns no bytes of that logical object.
        Its completion is neutral in the all-owner intersection. Sharded
        objects use the replica rank in their keys, so a different Host owner
        can restore a page written by an earlier allocation.
        """
        keys = [self.object_key(h, int(g), int(o)) for g, _, h, o in pages]
        if self.ownership is None:
            ranges = [
                self.host_storage.host_block_range(int(g), int(b))
                for g, b, _, _ in pages
            ]
            return keys, [offset for offset, _ in ranges], [size for _, size in ranges]
        offsets: list[int | None] = [None] * len(pages)
        sizes = [
            self.host_storage.host_cache_block_bytes[int(g)] for g, _, _, _ in pages
        ]
        owned = self.ownership.owned_host_positions(
            [(int(g), int(b)) for g, b, _, _ in pages]
        )
        for position, (group, local_block) in owned:
            offsets[position], sizes[position] = self.host_storage.host_block_range(
                group, local_block
            )
        return keys, offsets, sizes

    def close(self) -> None:
        self.backend.close()
