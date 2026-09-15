from __future__ import annotations

import json
import shutil
from pathlib import Path

from .contracts import CampaignSpec, CandidateSpec
from .evaluator import evaluate, judge_digest
from .hardware import inventory
from .integrity import digest, read_json, write_json
from .process import Lease


def diagnose(store, candidate_id, *, output, kind="sanitizers"):
    campaign = CampaignSpec.model_validate(store.verify(read_json(store.root / "campaign.json")))
    if campaign.mode != "hopper" or not campaign.gpu_execution:
        raise RuntimeError("diagnostics require an enabled Hopper campaign")
    if judge_digest() != campaign.source_snapshot_digest:
        raise ValueError("framework changed after campaign freeze")
    executable = "compute-sanitizer" if kind == "sanitizers" else "ncu"
    if not shutil.which(executable):
        raise RuntimeError(f"required diagnostic unavailable: {executable}")
    row = next((r for r in store.rows() if r["id"] == candidate_id), None)
    if not row or row["state"] != "numerical_pass":
        raise ValueError("candidate requires an independent numerical pass")
    candidate = CandidateSpec.model_validate(json.loads(row["spec"]))
    receipt = store.receipt(row["receipt"])
    artifact = read_json(store.root / "attempts" / row["attempt"] / "artifact.json")
    if receipt["artifact_manifest_digest"] != digest({"digest": artifact["digest"], "metadata": artifact["metadata"]}):
        raise ValueError("launch manifest changed")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    tools = ("memcheck", "initcheck", "racecheck", "synccheck") if kind == "sanitizers" else ("ncu",)
    seconds = campaign.budget.worker_timeout_seconds * len(campaign.task.cases) * len(tools)
    lease_path = Path.home() / ".cache/terminus-autokernel/leases" / f"{campaign.gpu_uuid}.lock"
    with Lease(store.root / "controller.lock") as controller, Lease(lease_path) as gpu:
        runtime = inventory(campaign.gpu_uuid, campaign.power_limit_watts)
        reports = {}
        with store.activity(campaign.identity, campaign.budget, seconds=seconds, kind=kind) as attempt:
            for tool in tools:
                prefix = (["compute-sanitizer", "--tool", tool, "--error-exitcode", "99", "--target-processes", "all"]
                          if kind == "sanitizers" else ["ncu", "--set", "basic", "--csv"])
                checked = evaluate(candidate, campaign, artifact, attempt, output / tool, runtime,
                                   pass_fds=(controller.fd, gpu.fd), worker_prefix=prefix)
                reports[tool] = {"status": checked.gates["numerical"].status, "error": checked.error,
                                 "scope": "Instrumented execution; timings excluded from performance evidence"}
                inventory(campaign.gpu_uuid, campaign.power_limit_watts)
            report = {"candidate_digest": candidate_id, "artifact_digest": artifact["digest"],
                      "judge_digest": judge_digest(), "kind": kind, "checks": reports,
                      "status": "passed" if all(r["status"] == "passed" for r in reports.values()) else "failed"}
            write_json(output / "report.json", store.envelope(report))
            store.event({"type": "diagnostic", "report": report})
        return report
