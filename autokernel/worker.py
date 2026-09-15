"""Trusted execution subprocess. No reference outputs or supervisor keys are passed here."""
from __future__ import annotations

import argparse
import ctypes as ct
import time
from pathlib import Path

import numpy as np

from .contracts import Case
from .integrity import file_digest, read_json, write_json


def execute(request):
    root = Path(request["directory"])
    case = Case.model_validate(request["case"])
    if file_digest(request["artifact"]["path"]) != request["artifact"]["digest"]:
        raise ValueError("artifact changed before execution")
    with np.load(root / "inputs.npz", allow_pickle=False) as data:
        inputs = {name: data[name].copy() for name in data.files}
    durations, device_ms = [], []
    driver = None
    if request["backend"] == "numpy_demo":
        program = read_json(request["artifact"]["path"])
        output = np.full((case.rows, case.columns), np.nan, dtype=np.float32)
        a = inputs["input"][:, :case.columns * case.column_stride:case.column_stride]
        b = inputs["rhs"][:, :case.columns * case.column_stride:case.column_stride]
        chunk = request["parameters"]["chunk"]

        def operation():
            for row in range(case.rows):
                for start in range(0, case.columns, chunk):
                    target = slice(start, start + chunk)
                    if program["operation"] == "add":
                        np.add(a[row, target], b[row, target], out=output[row, target])
                    elif program["operation"] == "subtract":
                        np.subtract(a[row, target], b[row, target], out=output[row, target])
                    elif program["operation"] == "mutate":
                        a[row, target] = 0
                        output[row, target] = b[row, target]

        def synchronize():
            pass
    else:
        from .cuda import Driver, quantizer_args

        driver = Driver()
        metadata = request["artifact"]["metadata"]
        function = driver.load(Path(request["artifact"]["path"]).read_bytes(), metadata["symbol"], metadata["shared"])
        input_ptr = driver.upload(inputs["input"])
        code_ptr = driver.upload(np.full((case.rows, case.columns), 0xA5, dtype=np.uint8))
        scale_shape = (case.columns // 128, (case.rows + 3) // 4 * 4)
        scale_ptr = driver.upload(np.full(scale_shape, np.nan, dtype=np.float32))
        arguments = quantizer_args(input_ptr, code_ptr, scale_ptr, case)
        arguments += [ct.c_uint64(0) for _ in range(metadata["extra_pointer_args"])]
        rows_per_cta = metadata["rows_per_cta"]

        def operation():
            driver.launch(function, arguments,
                          grid=((case.rows + rows_per_cta - 1) // rows_per_cta, case.columns // 128, 1),
                          block=metadata["block"], shared=metadata["shared"])

        synchronize = driver.synchronize
    try:
        for _ in range(5):
            operation()
        synchronize()
        for _ in range(request["repetitions"]):
            if driver:
                start_event, end_event = driver.event(), driver.event()
            started = time.perf_counter()
            if driver:
                driver.record(start_event)
            operation()
            if driver:
                driver.record(end_event)
            synchronize()
            durations.append(time.perf_counter() - started)
            if driver:
                device_ms.append(driver.elapsed(start_event, end_event))
                driver.check(driver.lib.cuEventDestroy_v2(start_event))
                driver.check(driver.lib.cuEventDestroy_v2(end_event))
        if driver:
            output = {"codes": driver.download(code_ptr, (case.rows, case.columns), np.uint8),
                      "scales": driver.download(scale_ptr, scale_shape, np.float32)[:, :case.rows].T.copy()}
            inputs["input"] = driver.download(input_ptr, inputs["input"].shape, inputs["input"].dtype)
        else:
            output = {"output": output}
        np.savez(root / "outputs.npz", **output)
        np.savez(root / "post_inputs.npz", **inputs)
        write_json(root / "result.json", {"host_seconds": durations, "device_ms": device_ms,
                                          "input_digest": file_digest(root / "inputs.npz"),
                                          "output_digest": file_digest(root / "outputs.npz")})
    finally:
        if driver:
            driver.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    execute(read_json(args.request))


if __name__ == "__main__":
    main()
