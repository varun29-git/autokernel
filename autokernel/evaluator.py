from __future__ import annotations

import secrets
import sys
from pathlib import Path

import numpy as np

from . import oracles
from .contracts import EvaluationReceipt, Gate
from .integrity import digest, file_digest, read_json, snapshot, write_json
from .process import clean_env, run_process


def judge_digest():
    return snapshot(Path(__file__).parent)["digest"]


def evaluate(candidate, campaign, artifact, attempt, directory, runtime, *, pass_fds=(), worker_prefix=()):
    task = campaign.task
    if file_digest(oracles.__file__) != task.reference_digest:
        raise ValueError("independent oracle changed after task freeze")
    records = []
    error = None
    try:
        for index, case in enumerate(task.cases):
            location = Path(directory) / f"case-{index}"
            location.mkdir(parents=True, exist_ok=True)
            seed = secrets.randbits(63)
            inputs = oracles.fixture(case, task.operation, seed)
            expected = oracles.reference(case, task.operation, inputs)
            np.savez(location / "inputs.npz", **inputs)
            request = {"directory": str(location), "case": case.model_dump(mode="json"),
                       "artifact": artifact, "backend": candidate.backend, "parameters": candidate.parameters,
                       "repetitions": campaign.timing_repetitions}
            write_json(location / "request.json", request)
            env = clean_env({"CUDA_VISIBLE_DEVICES": campaign.gpu_uuid} if campaign.gpu_uuid else {})
            run_process([*worker_prefix, sys.executable, "-m", "autokernel.worker", "--request", str(location / "request.json")],
                        cwd=location, timeout=campaign.budget.worker_timeout_seconds,
                        log_path=location / "worker.log", env=env, pass_fds=pass_fds)
            result = read_json(location / "result.json")
            if result["input_digest"] != file_digest(location / "inputs.npz"):
                raise AssertionError("worker changed input artifact")
            if result["output_digest"] != file_digest(location / "outputs.npz"):
                raise AssertionError("worker output artifact changed")
            with np.load(location / "post_inputs.npz", allow_pickle=False) as post:
                if set(post.files) != set(inputs):
                    raise AssertionError("input set changed")
                for key, value in inputs.items():
                    if value.dtype != post[key].dtype or value.shape != post[key].shape or value.tobytes() != post[key].tobytes():
                        raise AssertionError("kernel mutated a read-only input")
            with np.load(location / "outputs.npz", allow_pickle=False) as outputs:
                metrics = oracles.compare(dict(outputs), expected, task.numerical)
            if len(result["host_seconds"]) != campaign.timing_repetitions:
                raise AssertionError("missing timing samples")
            if any(not np.isfinite(t) or t <= 0 for t in result["host_seconds"]):
                raise AssertionError("invalid timing samples")
            if candidate.backend != "numpy_demo" and (
                len(result["device_ms"]) != campaign.timing_repetitions
                or any(not np.isfinite(t) or t <= 0 for t in result["device_ms"])
            ):
                raise AssertionError("invalid device timing samples")
            records.append({"case": case.model_dump(mode="json"), "seed": seed, "metrics": metrics,
                            **result})
            # Reproducible fixtures are represented by seed, case and digest. Keep failing arrays.
            for name in ("inputs.npz", "outputs.npz", "post_inputs.npz"):
                (location / name).unlink()
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    gates = {"numerical": Gate(status="failed" if error else "passed", details={"cases_passed": len(records)}),
             "sanitizers": Gate(status="pending"), "integration": Gate(status="pending"),
             "performance_confirmation": Gate(status="pending"), "quality": Gate(status="pending")}
    return EvaluationReceipt(campaign_digest=campaign.identity, candidate_digest=candidate.identity,
                             task_digest=task.identity, judge_digest=judge_digest(), artifact_digest=artifact["digest"],
                             artifact_manifest_digest=digest({"digest": artifact["digest"], "metadata": artifact["metadata"]}),
                             attempt_id=attempt, scope="cpu_demo" if candidate.backend == "numpy_demo" else "gpu_microbenchmark",
                             gates=gates, measurements={"cases": records}, runtime=runtime, error=error)
