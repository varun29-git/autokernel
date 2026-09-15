from __future__ import annotations

import contextlib
import fcntl
import os
import signal
import subprocess
import time
from pathlib import Path


class Lease:
    """Kernel-owned flock: time expiry never permits stealing a live GPU worker's lease."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            self.stream = None
            raise RuntimeError(f"lease is already held: {self.path}") from None
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(f"pid={os.getpid()} started={time.time()}\n")
        self.stream.flush()
        return self

    def __exit__(self, *args):
        fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.stream.close()

    @property
    def fd(self):
        return self.stream.fileno()


def clean_env(extra=None):
    keys = ("PATH", "HOME", "SYSTEMROOT", "TMPDIR", "LD_LIBRARY_PATH", "CUDA_HOME")
    env = {key: os.environ[key] for key in keys if key in os.environ}
    env.update({"PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    env.update(extra or {})
    return env


def run_process(argv, *, cwd, timeout, log_path, env=None, pass_fds=(), max_log_bytes=8 * 1024 * 1024,
                stdin_path=None):
    """No shell. All descendants killed on timeout; a timeout can never yield a valid result."""
    started = time.monotonic()
    with contextlib.ExitStack() as stack:
        log = stack.enter_context(Path(log_path).open("wb"))
        stdin = stack.enter_context(Path(stdin_path).open("rb")) if stdin_path else subprocess.DEVNULL
        process = subprocess.Popen(argv, cwd=cwd, env=env or clean_env(), stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True, pass_fds=pass_fds, stdin=stdin)
        try:
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(argv, timeout)
                if Path(log_path).stat().st_size > max_log_bytes:
                    raise RuntimeError("worker log exceeded its resource limit")
                try:
                    code = process.wait(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        finally:
            # A launcher exiting must not leave background children retaining leases or doing work.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if code:
        raise RuntimeError(f"worker exited {code}; log: {log_path}")
    return time.monotonic() - started
