from __future__ import annotations

import json
import time
from pathlib import Path

from .build import compile_candidate
from .contracts import CandidateSpec, EvaluationReceipt, Gate
from .evaluator import evaluate, judge_digest
from .hardware import inventory
from .integrity import snapshot, write_json
from .process import Lease
from .search import rank_candidates
from .telemetry import Telemetry


def run_campaign(store, campaign, candidates):
    store.freeze(campaign)
    if snapshot(Path(__file__).parent)["digest"] != campaign.source_snapshot_digest:
        raise ValueError("framework source changed after campaign freeze")
    if campaign.mode == "hopper" and not campaign.gpu_execution:
        raise RuntimeError("GPU execution is disabled; create a new explicitly enabled campaign")
    with Lease(store.root / "controller.lock") as controller_lease:
        gpu_path = Path.home() / ".cache/terminus-autokernel/leases" / f"{campaign.gpu_uuid}.lock"
        with Lease(gpu_path if campaign.mode == "hopper" else store.root / "demo.lock") as device_lease:
            runtime = inventory(campaign.gpu_uuid if campaign.mode == "hopper" else None,
                                campaign.power_limit_watts)
            store.recover()
            for candidate in candidates:
                if candidate.task_digest != campaign.task.identity:
                    raise ValueError("candidate does not match frozen task")
                if (candidate.backend == "numpy_demo") != (campaign.mode == "demo"):
                    raise ValueError("candidate backend does not match campaign mode")
                store.get(candidate.source_digest)
                store.enqueue(campaign.identity, candidate)
            pending = [CandidateSpec.model_validate(json.loads(row["spec"])) for row in store.rows()
                       if row["state"] == "proposed"]
            for candidate in rank_candidates(pending, beam_families=campaign.beam_families):
                if judge_digest() != campaign.source_snapshot_digest:
                    raise ValueError("framework changed during the frozen campaign")
                if store.storage_bytes() >= campaign.budget.max_storage_bytes:
                    store.event({"type": "budget_stop", "reason": "storage"})
                    break
                reservation = campaign.budget.worker_timeout_seconds * len(campaign.task.cases)
                if candidate.backend != "numpy_demo":
                    reservation += campaign.budget.build_timeout_seconds
                try:
                    attempt = store.claim(campaign.identity, candidate.identity, campaign.budget, reservation)
                except RuntimeError as error:
                    store.event({"type": "budget_stop", "reason": str(error)})
                    break
                started = time.monotonic()
                directory = store.root / "attempts" / attempt
                directory.mkdir(parents=True)
                try:
                    artifact = compile_candidate(candidate, campaign, store.get(candidate.source_digest), directory,
                                                 pass_fds=(controller_lease.fd, device_lease.fd))
                    write_json(directory / "artifact.json", artifact)
                    with Telemetry(campaign.gpu_uuid, directory / "telemetry.json"):
                        receipt = evaluate(candidate, campaign, artifact, attempt, directory, runtime,
                                           pass_fds=(controller_lease.fd, device_lease.fd))
                    if receipt.judge_digest != campaign.source_snapshot_digest:
                        raise ValueError("judge changed during evaluation")
                except Exception as error:
                    receipt = EvaluationReceipt(campaign_digest=campaign.identity, candidate_digest=candidate.identity,
                        task_digest=campaign.task.identity, judge_digest=judge_digest(), attempt_id=attempt,
                        scope="cpu_demo" if campaign.mode == "demo" else "gpu_microbenchmark",
                        gates={"numerical": Gate(status="failed")}, runtime=runtime,
                        error=f"{type(error).__name__}: {error}")
                store.finish(candidate.identity, attempt, receipt, time.monotonic() - started)
                # Recheck health/exclusivity after every native subprocess, including failures.
                if campaign.mode == "hopper":
                    inventory(campaign.gpu_uuid, campaign.power_limit_watts)
    return campaign_report(store)


def campaign_report(store):
    rows = []
    for row in store.rows():
        entry = {"candidate": row["id"], "state": row["state"], "receipt": row["receipt"]}
        if row["receipt"]:
            receipt = store.receipt(row["receipt"])
            entry["scope"] = receipt["scope"]
            entry["error"] = receipt["error"]
            cases = receipt["measurements"].get("cases", [])
            if cases:
                entry["exploratory_host_ms_sum"] = sum(
                    sum(case["host_seconds"]) / len(case["host_seconds"]) * 1000 for case in cases)
        rows.append(entry)
    correct = [r for r in rows if r["state"] == "numerical_pass" and "exploratory_host_ms_sum" in r]
    return {"candidates": rows,
            "best_exploratory_candidate": min(correct, key=lambda r: r["exploratory_host_ms_sum"])["candidate"] if correct else None,
            "release_status": "not_qualified",
            "scope": "Exploratory kernel search. No whole-training performance, GPU quality or production claim."}
