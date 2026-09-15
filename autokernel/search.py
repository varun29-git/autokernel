from __future__ import annotations

import itertools
import json
import os
import shutil
import tempfile
from pathlib import Path

from .contracts import CandidateSpec
from .integrity import canonical
from .process import run_process


def seed_candidates(store, campaign, *, include_wrong=False):
    task = campaign.task
    if campaign.mode == "demo":
        for operation in (["add", "subtract", "noop", "mutate"] if include_wrong else ["add"]):
            source = store.put(canonical({"operation": operation}))
            for chunk in (32, 256, 4096):
                yield CandidateSpec(task_digest=task.identity, backend="numpy_demo", family=f"chunked_{operation}",
                                    source_digest=source, parameters={"chunk": chunk},
                                    hypothesis="Measure chunk scheduling with an independent vector-add oracle.")
    else:
        templates = Path(__file__).parent / "templates"
        for backend, extension in (("cuda", "cu"), ("triton", "py")):
            source = store.put((templates / f"fp8_rows.{extension}").read_bytes())
            parameters = ([{"block": value} for value in (128, 256, 512)] if backend == "cuda" else
                          [{"num_warps": warps, "rows_per_cta": rows}
                           for warps, rows in itertools.product((4, 8), (1, 4, 8, 16))])
            for config in parameters:
                yield CandidateSpec(task_digest=task.identity, backend=backend, family=f"{backend}_row_groups",
                                    source_digest=source, parameters=config,
                                    hypothesis="Tune 128-value FP8 groups while preserving exact scale and code semantics.")


def rank_candidates(candidates, *, beam_families=4):
    """Round-robin families prevents one large parameter grid consuming the entire budget."""
    groups = {}
    for candidate in sorted(candidates, key=lambda c: (-c.expected_saved_ms, c.identity)):
        groups.setdefault(candidate.family, []).append(candidate)
    ordered = list(groups)
    result = []
    while ordered:
        for family in ordered[:beam_families]:
            result.append(groups[family].pop(0))
        ordered = [name for name in ordered if groups[name]]
        if ordered:
            offset = min(beam_families, len(ordered))
            ordered = ordered[offset:] + ordered[:offset]
    return result


def command_proposal(argv, task, observations, *, timeout=120, max_bytes=1_000_000):
    """Explicitly configured generator command; never executed from candidate metadata.

    Input: task + observations JSON. Output: {candidate, source}. Provider credentials
    belong to this generator process and never reach build/evaluation workers.
    """
    argv = list(argv)
    executable = Path(argv[0]).expanduser()
    has_path_component = os.sep in argv[0] or (os.altsep is not None and os.altsep in argv[0])
    if has_path_component:
        executable = executable if executable.is_absolute() else Path.cwd() / executable
        if not executable.is_file():
            raise FileNotFoundError(f"generator executable does not exist: {executable}")
        argv[0] = str(executable.resolve())
    else:
        resolved = shutil.which(argv[0])
        if not resolved:
            raise FileNotFoundError(f"generator executable is not on PATH: {argv[0]}")
        argv[0] = resolved
    request = canonical({"task": task.model_dump(mode="json"), "observations": observations,
                         "candidate_schema": CandidateSpec.model_json_schema()})
    if len(request) > max_bytes:
        raise ValueError("generator request exceeded its resource limit")
    with tempfile.TemporaryDirectory(prefix="autokernel-generator-") as directory:
        root = Path(directory)
        (root / "request.json").write_bytes(request)
        run_process(argv, cwd=root, timeout=timeout, log_path=root / "response.json",
                    stdin_path=root / "request.json", env=dict(os.environ), max_log_bytes=max_bytes)
        response = (root / "response.json").read_bytes()
        if len(response) > max_bytes:
            raise ValueError("oversized generator response")
        payload = json.loads(response)
    if set(payload) != {"candidate", "source"} or not isinstance(payload["source"], str):
        raise ValueError("invalid generator response")
    candidate = CandidateSpec.model_validate(payload["candidate"])
    if candidate.task_digest != task.identity:
        raise ValueError("generator changed frozen task")
    import hashlib

    source = payload["source"].encode()
    if hashlib.sha256(source).hexdigest() != candidate.source_digest:
        raise ValueError("generator source digest mismatch")
    return candidate, source
