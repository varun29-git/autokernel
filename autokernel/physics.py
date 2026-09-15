from __future__ import annotations

import math


def resource_bound(*, fp8_flops=0, bf16_flops=0, compulsory_bytes=0,
                   fp8_tflops=1671.0, bf16_tflops=835.5, bandwidth_tb_s=3.9):
    values = (fp8_flops, bf16_flops, compulsory_bytes, fp8_tflops, bf16_tflops, bandwidth_tb_s)
    if any(not math.isfinite(v) or v < 0 for v in values) or min(values[3:]) <= 0:
        raise ValueError("invalid resource accounting")
    compute = fp8_flops / (fp8_tflops * 1e12) + bf16_flops / (bf16_tflops * 1e12)
    memory = compulsory_bytes / (bandwidth_tb_s * 1e12)
    return {"optimistic_compute_ms": compute * 1000, "optimistic_memory_ms": memory * 1000,
            "optimistic_resource_bound_ms": max(compute, memory) * 1000,
            "scope": "Advertised dense single-GPU peaks; assumed shared tensor-core resource. "
                     "Not empirical sustained performance or an algorithm-independent optimum."}


def target_gap(tokens, update_seconds, target):
    if min(tokens, update_seconds, target) <= 0 or not all(math.isfinite(x) for x in (tokens, update_seconds, target)):
        raise ValueError("invalid throughput target")
    target_seconds = tokens / target
    return {"tokens_per_second": tokens / update_seconds, "target_update_ms": target_seconds * 1000,
            "remaining_ms": max(0, update_seconds - target_seconds) * 1000,
            "required_throughput_factor": update_seconds / target_seconds}


def pareto_front(rows):
    eligible = [r for r in rows if r["correct"] and math.isfinite(r["milliseconds"]) and r["milliseconds"] > 0]
    return [row for row in eligible if not any(
        other["milliseconds"] <= row["milliseconds"] and other["workspace_bytes"] <= row["workspace_bytes"]
        and (other["milliseconds"] < row["milliseconds"] or other["workspace_bytes"] < row["workspace_bytes"])
        for other in eligible)]
