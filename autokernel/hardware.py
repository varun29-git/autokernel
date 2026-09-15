from __future__ import annotations

import csv
import io
import platform
import subprocess
import sys

from .integrity import digest


def inventory(gpu_uuid=None, expected_power=400.0):
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "executable": sys.executable, "gpu": None}
    if gpu_uuid:
        columns = "uuid,name,power.limit,memory.total,driver_version"
        text = subprocess.check_output(["nvidia-smi", f"--id={gpu_uuid}", f"--query-gpu={columns}",
                                        "--format=csv,noheader,nounits"], text=True, timeout=10)
        rows = list(csv.reader(io.StringIO(text)))
        if len(rows) != 1 or len(rows[0]) != 5:
            raise RuntimeError("ambiguous GPU inventory")
        uuid, name, power, memory, driver = [v.strip() for v in rows[0]]
        if uuid != gpu_uuid or "H100" not in name or "NVL" not in name:
            raise RuntimeError("GPU identity differs from frozen H100 NVL campaign")
        if float(power) != expected_power:
            raise RuntimeError("GPU power policy changed")
        active = subprocess.check_output(["nvidia-smi", f"--id={gpu_uuid}",
                                          "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                         text=True, timeout=10).strip()
        if active:
            raise RuntimeError("GPU has another active compute process")
        result["gpu"] = {"uuid": uuid, "name": name, "power_limit_watts": float(power),
                          "memory_mib": int(memory), "driver": driver}
    result["digest"] = digest(result)
    return result
