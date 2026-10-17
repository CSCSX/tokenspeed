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

"""Launcher cleanup after real worker failures and termination signals."""

import contextlib
import functools
import multiprocessing as mp
import os
import signal
import sys
import threading

import psutil
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.utils.process import (  # noqa: E402
    register_child_failure_signals,
    start_process_with_ready_pipe,
)

_TIMEOUT_S = 60


def _wait_for_termination(writer):
    writer.send({"status": "ready"})
    threading.Event().wait(_TIMEOUT_S * 10)


def _engine_startup_failure(signum, writer):
    from tokenspeed.runtime.entrypoints import engine as engine_module
    from tokenspeed.runtime.utils.server_args import ServerArgs

    started = []

    def launch_without_model(server_args, port_args):
        # Replace GPU/model loading only; Engine's constructor, failure handler,
        # signal callbacks and process-tree shutdown are the real implementation.
        register_child_failure_signals()
        for _ in range(2):
            proc, reader = start_process_with_ready_pipe(
                _wait_for_termination, (), contextlib.nullcontext()
            )
            started.append((proc, reader))
            assert reader.poll(_TIMEOUT_S)
            assert reader.recv() == {"status": "ready"}
        os.kill(os.getpid(), signum)
        raise AssertionError("child-failure signal did not unwind startup")

    args = ServerArgs(model="unused", device="cpu", world_size=1, nprocs_per_node=1)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(engine_module, "_launch_subprocesses", launch_without_model)
            try:
                engine_module.Engine(server_args=args)
            except SystemExit as exc:
                for proc, _ in started:
                    proc.join(_TIMEOUT_S)
                writer.send(
                    {
                        "status": exc.code,
                        "exitcodes": [proc.exitcode for proc, _ in started],
                        "alive": [proc.is_alive() for proc, _ in started],
                    }
                )
    finally:
        for proc, reader in started:
            if proc.is_alive():
                proc.kill()
            proc.join(_TIMEOUT_S)
            reader.close()
        writer.close()


def _dp_model_start(
    mode, startup_barrier, status_writer, server_args, port_args, writer
):
    rank = server_args.mapping.attn.dp_rank
    startup_barrier.wait(_TIMEOUT_S)
    status_writer.send((rank, os.getpid()))
    startup_barrier.wait(_TIMEOUT_S)
    if mode == "failed_group" and rank == 0:
        # Failure while the other group is still in startup. A lingering thread
        # can hold the process alive even after its ready channel closes.
        writer.close()
    threading.Event().wait(_TIMEOUT_S * 10)


def _dp_controller(mode, result_writer, writer):
    from tokenspeed.runtime.engine import data_parallel_controller as controller_module
    from tokenspeed.runtime.engine import event_loop as event_loop_module
    from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs

    status_reader, status_writer = mp.Pipe(duplex=False)
    startup_barrier = mp.Barrier(2)
    worker_pids = []

    def observe_workers():
        for _ in range(2):
            if not status_reader.poll(_TIMEOUT_S):
                return
            worker_pids.append(status_reader.recv()[1])
        if mode == "terminate":
            os.kill(os.getpid(), signal.SIGTERM)

    args = ServerArgs(
        model="unused",
        device="cpu",
        world_size=2,
        nprocs_per_node=2,
        tensor_parallel_size=1,
        data_parallel_size=2,
        dist_init_addr="127.0.0.1:28000",
    )
    ports = PortArgs.init_new(args)
    observer = threading.Thread(target=observe_workers, daemon=True)
    observer.start()
    # Use real DP launch, readiness supervision and wrapper teardown; replace
    # only GPU/model loading with two real processes in a startup collective.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            event_loop_module,
            "run_event_loop",
            functools.partial(_dp_model_start, mode, startup_barrier, status_writer),
        )
        try:
            controller_module.run_data_parallel_controller_process(args, ports, writer)
        except (RuntimeError, SystemExit) as exc:
            outcome = str(exc) if isinstance(exc, RuntimeError) else exc.code
        else:
            outcome = "returned without failure"
    observer.join(_TIMEOUT_S)
    assert not observer.is_alive(), "worker observer did not finish"
    assert len(worker_pids) == 2, "both DP groups must start before the failure"
    # The wrapper owns termination although its constructor never returned.
    # Joining here reaps children; it must not be needed to stop them.
    for child in mp.active_children():
        child.join(_TIMEOUT_S)
    result_writer.send(
        {
            "outcome": outcome,
            "alive": [psutil.pid_exists(pid) for pid in worker_pids],
            "ready_closed": writer.closed,
        }
    )
    status_reader.close()
    status_writer.close()
    result_writer.close()


def _dp_supervisor(mode, result_writer, writer):
    # Consume the controller's real child-failure notification in an isolated
    # parent, rather than delivering it to pytest itself.
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    proc = mp.Process(target=_dp_controller, args=(mode, result_writer, writer))
    proc.start()
    writer.close()
    result_writer.close()
    proc.join(_TIMEOUT_S)
    assert not proc.is_alive(), "DP controller did not finish cleanup"
    assert proc.exitcode == 0


def _stop_supervisor(proc):
    if proc.is_alive():
        # Test-only fallback must stop descendants if the regression hangs.
        try:
            for child in psutil.Process(proc.pid).children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            pass
        proc.kill()
    proc.join(_TIMEOUT_S)


@pytest.mark.parametrize("signum", [signal.SIGUSR1, signal.SIGQUIT])
def test_engine_startup_signal_owns_worker_cleanup(signum):
    proc, reader = start_process_with_ready_pipe(
        _engine_startup_failure, (signum,), contextlib.nullcontext()
    )
    try:
        assert reader.poll(_TIMEOUT_S), "Engine did not finish startup cleanup"
        result = reader.recv()
        assert result["status"] == 1
        assert result["alive"] == [False, False]
        assert result["exitcodes"] == [-signal.SIGKILL, -signal.SIGKILL]
        proc.join(_TIMEOUT_S)
        assert proc.exitcode == 0
    finally:
        _stop_supervisor(proc)
        reader.close()


@pytest.mark.parametrize("mode", ["failed_group", "terminate"])
def test_dp_partial_startup_owns_worker_cleanup(mode):
    result_reader, result_writer = mp.Pipe(duplex=False)
    proc, ready_reader = start_process_with_ready_pipe(
        _dp_supervisor, (mode, result_writer), contextlib.nullcontext()
    )
    result_writer.close()
    try:
        assert result_reader.poll(_TIMEOUT_S), "DP controller did not unwind startup"
        result = result_reader.recv()
        if mode == "failed_group":
            assert "Scheduler process 0" in result["outcome"]
        else:
            assert result["outcome"] == 0
        assert result["alive"] == [False, False]
        assert result["ready_closed"]
        assert ready_reader.poll(_TIMEOUT_S)
        with pytest.raises(EOFError):
            ready_reader.recv()
        proc.join(_TIMEOUT_S)
        assert proc.exitcode == 0
    finally:
        _stop_supervisor(proc)
        ready_reader.close()
        result_reader.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
