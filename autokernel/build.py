from __future__ import annotations

import re
import secrets
import shutil
import subprocess
from pathlib import Path

from .integrity import atomic_write, canonical, file_digest, read_json
from .process import clean_env, run_process


def compile_candidate(candidate, campaign, source, directory, *, pass_fds=()):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if candidate.backend == "numpy_demo":
        import json

        program = json.loads(source)
        if set(program) != {"operation"} or program["operation"] not in {"add", "subtract", "noop", "mutate"}:
            raise ValueError("invalid constrained demo program")
        atomic_write(directory / "kernel.json", canonical(program))
        return {"path": str(directory / "kernel.json"), "digest": file_digest(directory / "kernel.json"),
                "metadata": {"backend": "numpy_demo"}}
    if not campaign.gpu_execution or campaign.mode != "hopper":
        raise RuntimeError("native execution disabled in frozen campaign")
    if not shutil.which("docker"):
        raise RuntimeError("Docker unavailable; unsafe host compilation is deliberately unsupported")
    if not re.fullmatch(r"[A-Za-z0-9._/:\-]+@sha256:[a-f0-9]{64}", campaign.build_image or ""):
        raise ValueError("native build requires an immutable container image digest")
    inputs, outputs = directory / "input", directory / "output"
    inputs.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)
    # The unprivileged container user must be able to write only this scratch output directory.
    outputs.chmod(0o777)
    extension = "cu" if candidate.backend == "cuda" else "py"
    atomic_write(inputs / f"kernel.{extension}", source, mode=0o644)
    atomic_write(inputs / "build_worker.py", Path(__file__).with_name("build_worker.py").read_bytes(), mode=0o644)
    name = "autokernel-build-" + secrets.token_hex(8)
    argv = ["docker", "run", "--rm", "--name", name, "--network=none", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=65534:65534",
            "--pids-limit=128", "--memory=8g", "--cpus=4", "--ulimit=fsize=67108864:67108864",
            "--tmpfs=/tmp:rw,nosuid,nodev,size=2147483648", "--env=HOME=/tmp",
            "--env=TRITON_CACHE_DIR=/tmp/triton", "--mount", f"type=bind,src={inputs},dst=/input,readonly",
            "--mount", f"type=bind,src={outputs},dst=/output", campaign.build_image,
            "timeout", "--signal=KILL", str(int(campaign.budget.build_timeout_seconds)),
            "python", "/input/build_worker.py", "--backend", candidate.backend,
            "--parameters", canonical(candidate.parameters).decode()]
    try:
        run_process(argv, cwd=directory, timeout=campaign.budget.build_timeout_seconds,
                    log_path=directory / "build.log", pass_fds=pass_fds)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15, env=clean_env(), check=False)
    for path in (outputs / "metadata.json", outputs / "kernel.cubin"):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("invalid or oversized build artifact")
    metadata = read_json(outputs / "metadata.json")
    if set(metadata) != {"symbol", "block", "shared", "rows_per_cta", "extra_pointer_args"}:
        raise ValueError("invalid artifact metadata")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,255}", metadata["symbol"]):
        raise ValueError("invalid kernel symbol")
    if metadata["block"] not in (128, 256, 512) or not 0 <= metadata["shared"] <= 227 * 1024:
        raise ValueError("invalid launch resources")
    expected_block = candidate.parameters.get("block", candidate.parameters.get("num_warps", 4) * 32)
    if metadata["block"] != expected_block or metadata["extra_pointer_args"] not in (0, 1, 2):
        raise ValueError("compiled launch ABI differs from candidate contract")
    expected_rows = candidate.parameters.get("rows_per_cta", 1)
    if metadata["rows_per_cta"] != expected_rows:
        raise ValueError("unexpected grid contract")
    # Copy out before execution; subsequent container writes cannot change the loaded bytes.
    target = directory / "kernel.cubin"
    binary = (outputs / "kernel.cubin").read_bytes()
    if len(binary) < 64 or binary[:6] != b"\x7fELF\x02\x01" or int.from_bytes(binary[18:20], "little") != 190:
        raise ValueError("artifact is not a CUDA ELF cubin")
    atomic_write(target, binary)
    return {"path": str(target), "digest": file_digest(target), "metadata": metadata}
