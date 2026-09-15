"""Trusted full-training bridge. Run only on the explicitly selected H100 host."""
from __future__ import annotations

import argparse
import ctypes as ct
import sys
from pathlib import Path

from .integrity import file_digest, read_json, snapshot, write_json


def install_quantizer(artifact):
    import terminus.fp8_kernels as kernels
    import torch

    from .cuda import Driver

    torch.cuda.init()
    driver = Driver(create_context=False)
    metadata = artifact["metadata"]
    if file_digest(artifact["path"]) != artifact["digest"]:
        raise ValueError("candidate artifact changed")
    function = driver.load(Path(artifact["path"]).read_bytes(), metadata["symbol"], metadata["shared"])
    original = kernels.quantize_directions
    counts = {"candidate_calls": 0, "fallback_calls": 0}

    def quantize(values, *, rowwise=True, transposed=False, eps=1e-5, separate=False):
        if (not rowwise or values.dtype != torch.bfloat16 or values.ndim != 2 or
            not 1 <= values.shape[0] <= 32768 or values.shape[1] % 128 or values.shape[1] > 10752 or
            values.stride(0) <= 0 or values.stride(1) <= 0):
            counts["fallback_calls"] += 1
            return original(values, rowwise=rowwise, transposed=transposed, eps=eps, separate=separate)
        secondary = original(values, rowwise=False, transposed=True, eps=eps, separate=separate)[1] if transposed else None
        rows, width = values.shape
        stride = (rows + 3) // 4 * 4
        codes = torch.empty((rows, width), dtype=torch.float8_e4m3fn, device=values.device)
        scales = torch.empty_strided((rows, width // 128), (1, stride), dtype=torch.float32, device=values.device)
        args = [ct.c_uint64(values.data_ptr()), ct.c_uint64(codes.data_ptr()), ct.c_uint64(scales.data_ptr()),
                ct.c_int64(rows), ct.c_int64(width), ct.c_int64(values.stride(0)), ct.c_int64(values.stride(1)),
                ct.c_int64(stride), ct.c_float(eps)]
        args += [ct.c_uint64(0) for _ in range(metadata["extra_pointer_args"])]
        tile = metadata["rows_per_cta"]
        driver.launch(function, args, grid=((rows + tile - 1) // tile, width // 128, 1), block=metadata["block"],
                      shared=metadata["shared"], stream=torch.cuda.current_stream(values.device).cuda_stream)
        counts["candidate_calls"] += 1
        return (codes, scales), secondary

    kernels.quantize_directions = quantize
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    repository = Path(request["repository"]).resolve()
    if snapshot(repository / "terminus")["digest"] != request["source_digest"]:
        raise ValueError("Terminus source changed")
    sys.path.insert(0, str(repository))
    import terminus.benchmark as benchmark
    import torch
    from terminus.preflight import run_preflight
    from terminus.train import load_config

    torch.set_num_threads(4)
    counts = install_quantizer(request["artifact"]) if request.get("artifact") else None
    config, training, _ = load_config(request["config"])
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preflight.json", run_preflight(config, torch.device("cuda", 0),
                                                       cuda_graphs=training.cuda_graphs))
    captured = {}
    model_factory, optimizer_factory = benchmark.BitNetLlama, benchmark.build_optimizer
    original_randint = torch.randint

    def capture_inputs(*args, **kwargs):
        tensor = original_randint(*args, **kwargs)
        expected = (training.gradient_accumulation_steps, request["batch"], config.max_seq_len + 1)
        if tuple(tensor.shape) == expected and tensor.device.type == "cuda":
            captured["inputs"] = tensor
            captured["input_pool"] = original_randint(config.vocab_size, (16, *expected), device=tensor.device)
            captured["input_index"] = 0
            torch.randint = original_randint
        return tensor

    def capture_model(*args, **kwargs):
        model = model_factory(*args, **kwargs)
        captured["model"] = model
        return model

    def capture_optimizer(*args, **kwargs):
        optimizer = optimizer_factory(*args, **kwargs)
        captured["optimizer"] = optimizer
        zero_grad = optimizer.zero_grad

        def rotate_then_zero(*args, **kwargs):
            if "inputs" in captured:
                index = captured["input_index"]
                captured["inputs"].copy_(captured["input_pool"][index % 16])
                captured["input_index"] = index + 1
            return zero_grad(*args, **kwargs)

        optimizer.zero_grad = rotate_then_zero
        return optimizer

    benchmark.BitNetLlama, benchmark.build_optimizer = capture_model, capture_optimizer
    torch.randint = capture_inputs
    receipt = benchmark.training_benchmark(request["config"], batch_size=request["batch"],
                 warmup_steps=request["warmup"], measured_steps=request["steps"],
                 preflight_report=output / "preflight.json")
    receipt["autokernel_adapter"] = {"artifact": request.get("artifact"), "calls": counts,
                                     "rotating_input_batches": 16,
                                     "scope": "Trusted FP8 row adapter; transposed operands use incumbent; "
                                              "rotating synthetic input copy included in update time"}
    if "inputs" not in captured:
        raise RuntimeError("training input rotation adapter did not bind; unsupported benchmark implementation")
    if counts and not counts["candidate_calls"]:
        raise RuntimeError("candidate was never executed")
    write_json(output / "benchmark.json", receipt)
    if request["save_state"]:
        # One tensor per file bounds comparison memory. Snapshot work is outside the timed window.
        states = output / "state"
        states.mkdir(exist_ok=True)
        manifest = {}

        def save(name, tensor):
            filename = f"{len(manifest):06d}.pt"
            torch.save(tensor.detach().cpu(), states / filename)
            manifest[name] = {"file": filename, "digest": file_digest(states / filename),
                              "shape": list(tensor.shape), "dtype": str(tensor.dtype)}

        for name, parameter in captured["model"].named_parameters():
            save("parameter/" + name, parameter)
            if parameter.grad is not None:
                save("gradient/" + name, parameter.grad)
            for key, value in captured["optimizer"].state.get(parameter, {}).items():
                if isinstance(value, torch.Tensor):
                    save(f"optimizer/{name}/{key}", value)
                else:
                    raise TypeError("unsupported optimizer-state type; do not silently omit state")
        for name, buffer in captured["model"].named_buffers():
            save("buffer/" + name, buffer)
        save("rng/cpu", torch.get_rng_state())
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            save(f"rng/cuda/{index}", state)
        if any(getattr(module, "_fp8_weight_cache", None) is not None
               for module in captured["model"].modules()):
            raise AssertionError("FP8 cache was not invalidated after optimizer update")
        write_json(states / "manifest.json", manifest)


if __name__ == "__main__":
    main()
