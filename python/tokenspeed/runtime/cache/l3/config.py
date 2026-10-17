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

"""Validated limits shared by the storage executor and C++ cache owner."""

from dataclasses import dataclass

from tokenspeed.runtime.cache.l3.mooncake import parse_extra_config


@dataclass(frozen=True, slots=True)
class L3AsyncConfig:
    max_operations: int
    max_keys_per_operation: int
    max_host_blocks: int
    metadata_entries: int
    max_prefetch_tokens: int
    min_prefetch_tokens: int
    wait_timeout_ms: int
    ready_timeout_ms: int
    query_workers: int
    read_workers: int
    write_workers: int
    failure_threshold: int
    recovery_interval_ms: int
    queue_high_operations: int
    queue_low_operations: int
    queue_high_bytes: int
    queue_low_bytes: int
    operation_high_age_ms: int
    operation_low_age_ms: int

    @classmethod
    def from_extra_config(
        cls, raw: str | None, *, host_blocks: int, prefix_granularity: int
    ):
        knobs = parse_extra_config(raw).get("async", {})
        if not isinstance(knobs, dict):
            raise ValueError("L3 async config must be an object")
        # Tuning values within the selected asynchronous backend, not switches
        # between schedulers or memory-safety implementations.
        values = dict(
            max_operations=32,
            max_keys_per_operation=256,
            max_host_blocks=max(1, min(256, host_blocks // 2)),
            metadata_entries=max(1, host_blocks * 4),
            max_prefetch_tokens=8192,
            min_prefetch_tokens=prefix_granularity,
            wait_timeout_ms=2000,
            ready_timeout_ms=1000,
            query_workers=2,
            read_workers=2,
            write_workers=1,
            failure_threshold=1,
            recovery_interval_ms=30000,
            queue_high_operations=8,
            queue_low_operations=4,
            queue_high_bytes=256 * 1024 * 1024,
            queue_low_bytes=128 * 1024 * 1024,
            operation_high_age_ms=500,
            operation_low_age_ms=250,
        )
        unknown = knobs.keys() - values.keys()
        if unknown:
            raise ValueError(f"Unsupported L3 async options: {sorted(unknown)}")
        values.update(knobs)
        for name, value in values.items():
            minimum = (
                0
                if name
                in {"queue_low_operations", "queue_low_bytes", "operation_low_age_ms"}
                else 1
            )
            if type(value) is not int or value < minimum or value > (2**63 - 1) // 1000:
                raise ValueError(
                    f"L3 {name} must be an integer in [{minimum}, {(2**63 - 1) // 1000}]"
                )
        if "queue_high_operations" not in knobs:
            values["queue_high_operations"] = min(8, values["max_operations"])
        if "queue_low_operations" not in knobs:
            values["queue_low_operations"] = values["queue_high_operations"] // 2
        for high, low in (
            ("queue_high_operations", "queue_low_operations"),
            ("queue_high_bytes", "queue_low_bytes"),
            ("operation_high_age_ms", "operation_low_age_ms"),
        ):
            if values[low] >= values[high]:
                raise ValueError(f"L3 {low} must be smaller than {high}")
        if values["max_host_blocks"] > host_blocks:
            raise ValueError("L3 max_host_blocks exceeds usable Host parents")
        if values["min_prefetch_tokens"] > values["max_prefetch_tokens"]:
            raise ValueError("L3 minimum prefetch must fit the maximum")
        return cls(**values)

    def scheduler_config(self):
        from tokenspeed_scheduler import StorageConfig, StoragePressure

        return StorageConfig(
            self.max_operations,
            self.max_keys_per_operation,
            self.max_host_blocks,
            self.metadata_entries,
            self.max_prefetch_tokens,
            self.min_prefetch_tokens,
            self.wait_timeout_ms * 1000,
            self.ready_timeout_ms * 1000,
            StoragePressure(
                self.queue_high_operations,
                self.queue_high_bytes,
                self.operation_high_age_ms * 1000,
            ),
            StoragePressure(
                self.queue_low_operations,
                self.queue_low_bytes,
                self.operation_low_age_ms * 1000,
            ),
        )
