from __future__ import annotations

import copy
import sys
from pathlib import Path

from .contracts import CampaignSpec
from .hardware import inventory
from .integrity import digest, file_digest, read_json, snapshot, write_json
from .process import Lease, clean_env, run_process
from .telemetry import Telemetry


def compare_states(left, right):
    import torch

    left, right = Path(left), Path(right)
    a, b = read_json(left / "manifest.json"), read_json(right / "manifest.json")
    if a.keys() != b.keys():
        raise AssertionError("state tensor set changed")
    mismatches = []
    for name in a:
        for directory, spec in ((left, a[name]), (right, b[name])):
            if Path(spec["file"]).name != spec["file"] or file_digest(directory / spec["file"]) != spec["digest"]:
                raise ValueError("state artifact changed")
        x = torch.load(left / a[name]["file"], map_location="cpu", weights_only=True)
        y = torch.load(right / b[name]["file"], map_location="cpu", weights_only=True)
        if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
            mismatches.append(name)
    return {"status": "passed" if not mismatches else "failed", "tensors_compared": len(a),
            "mismatches": mismatches, "policy": "exact final parameter/gradient/Adam-state equality"}


def training_pair(store, candidate_id, imported, *, output, batch=8, warmup=5, steps=8,
                  timeout=1800, save_state=True):
    campaign = CampaignSpec.model_validate(store.verify(read_json(store.root / "campaign.json")))
    if campaign.mode != "hopper" or not campaign.gpu_execution:
        raise RuntimeError("training requires an explicitly enabled Hopper campaign")
    if batch not in (1, 2, 4, 8) or steps < 1 or warmup < 0 or timeout <= 0:
        raise ValueError("invalid training comparison workload")
    row = next((r for r in store.rows() if r["id"] == candidate_id), None)
    if not row or row["state"] != "numerical_pass":
        raise ValueError("candidate needs a signed numerical pass before training integration")
    receipt = store.receipt(row["receipt"])
    artifact = read_json(store.root / "attempts" / row["attempt"] / "artifact.json")
    if receipt["artifact_digest"] != artifact["digest"] or file_digest(artifact["path"]) != artifact["digest"]:
        raise ValueError("artifact does not match judged candidate")
    if receipt["artifact_manifest_digest"] != digest({"digest": artifact["digest"], "metadata": artifact["metadata"]}):
        raise ValueError("candidate launch manifest changed")
    from .evaluator import judge_digest
    if judge_digest() != campaign.source_snapshot_digest:
        raise ValueError("framework changed after candidate evaluation")
    if snapshot(Path(imported["repository"]) / "terminus")["digest"] != imported["source"]["digest"]:
        raise ValueError("Terminus import is stale")
    if imported["resolved_config"]["model"] != campaign.immutable_model:
        raise ValueError("training architecture differs from campaign")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(imported["resolved_config"])
    config["training"]["gradient_accumulation_steps"] = 8 // batch
    write_json(output / "config.json", config)
    lease_path = Path.home() / ".cache/terminus-autokernel/leases" / f"{campaign.gpu_uuid}.lock"
    with Lease(store.root / "controller.lock") as control_lease, Lease(lease_path) as lease, \
         store.activity(campaign.identity, campaign.budget, seconds=2 * timeout, kind="training_pair"):
        before = inventory(campaign.gpu_uuid, campaign.power_limit_watts)
        for arm in ("control", "candidate"):
            request = {"repository": imported["repository"], "source_digest": imported["source"]["digest"],
                       "config": str(output / "config.json"), "batch": batch, "warmup": warmup, "steps": steps,
                       "output": str(output / arm), "save_state": save_state,
                       "artifact": artifact if arm == "candidate" else None}
            write_json(output / f"{arm}_request.json", request)
            with Telemetry(campaign.gpu_uuid, output / f"{arm}_telemetry.json"):
                run_process([sys.executable, "-m", "autokernel.training_worker", "--request",
                             str(output / f"{arm}_request.json")], cwd=output, timeout=timeout,
                            log_path=output / f"{arm}.log", env=clean_env({"CUDA_VISIBLE_DEVICES": campaign.gpu_uuid}),
                            pass_fds=(control_lease.fd, lease.fd))
            inventory(campaign.gpu_uuid, campaign.power_limit_watts)
        comparison = compare_states(output / "control/state", output / "candidate/state") if save_state else {
            "status": "pending", "reason": "state snapshot comparison disabled"}
        report = {"candidate_digest": candidate_id, "artifact_digest": artifact["digest"],
                  "judge_digest": judge_digest(), "runtime": before, "state_comparison": comparison,
                  "status": "state_comparison_pass" if comparison["status"] == "passed" else "integration_pending",
                  "scope": "Single paired integration check; independent performance confirmation, sanitizers, "
                           "checkpoint recovery and real-data quality are still required"}
        write_json(output / "comparison.json", store.envelope(report))
        store.event({"type": "training_pair", "report": report})
    return report
