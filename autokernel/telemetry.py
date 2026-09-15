"""Telemetry is collected by the supervisor, outside timed/candidate code."""
from __future__ import annotations

import subprocess
import threading
import time

from .integrity import write_json


class Telemetry:
    def __init__(self, uuid, path, interval=.5):
        self.uuid, self.path, self.interval = uuid, path, interval
        self.samples = []
        self.stop = threading.Event()
        self.thread = None

    def _collect(self):
        fields = "uuid,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used"
        while not self.stop.is_set():
            sample = {"unix_seconds": time.time(), "fields": fields.split(",")}
            try:
                output = subprocess.check_output(["nvidia-smi", f"--id={self.uuid}",
                          f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True, timeout=5)
                sample["values"] = [v.strip() for v in output.strip().split(",")]
            except (OSError, subprocess.SubprocessError) as error:
                sample["error"] = str(error)
            self.samples.append(sample)
            self.stop.wait(self.interval)

    def __enter__(self):
        if self.uuid:
            self.thread = threading.Thread(target=self._collect, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=6)
        write_json(self.path, {"samples": self.samples,
                              "scope": "Sampled mixed-workload telemetry; not per-kernel clocks or bandwidth"})
