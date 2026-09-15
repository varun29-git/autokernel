"""Collect a frozen, randomized, balanced full-training timing protocol on one leased H100."""
from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

from .analysis import TimingBlock
from .contracts import CampaignSpec
from .evaluator import judge_digest
from .hardware import inventory
from .integrity import digest, read_json, snapshot, write_json
from .process import Lease, clean_env, run_process
from .telemetry import Telemetry


def balanced_schedule(policy, seed):
    if policy.required_pairs % policy.required_sessions or policy.required_pairs % 2:
        raise ValueError("balanced collection needs an even pair count divisible by sessions")
    rng = random.Random(seed)
    orders = [("control", "candidate"), ("candidate", "control")] * (policy.required_pairs // 2)
    rng.shuffle(orders)
    per_session = policy.required_pairs // policy.required_sessions
    return [{"pair_id": str(index), "session": str(index // per_session), "order": list(order)}
            for index, order in enumerate(orders)]


def collect_training(store, candidate_id, imported, policy, *, output, integration_report,
                     sanitizer_report, aa=False, seed=741, warmup=32, timeout=600):
    campaign = CampaignSpec.model_validate(store.verify(read_json(store.root / "campaign.json")))
    if campaign.mode != "hopper" or not campaign.gpu_execution:
        raise RuntimeError("collection requires an enabled Hopper campaign")
    if judge_digest() != campaign.source_snapshot_digest:
        raise ValueError("framework changed after evaluation")
    state = store.verify(read_json(integration_report))
    sanitized = store.verify(read_json(sanitizer_report))
    for report in (state, sanitized):
        if report["candidate_digest"] != candidate_id or report["judge_digest"] != judge_digest():
            raise ValueError("qualification evidence belongs to a different candidate or judge")
    if state["status"] != "state_comparison_pass" or sanitized["status"] != "passed" or sanitized["kind"] != "sanitizers":
        raise ValueError("required state/sanitizer checks have not passed")
    row = next(r for r in store.rows() if r["id"] == candidate_id)
    artifact = read_json(store.root / "attempts" / row["attempt"] / "artifact.json")
    judged = store.receipt(row["receipt"])
    if judged["artifact_manifest_digest"] != digest({"digest": artifact["digest"], "metadata": artifact["metadata"]}):
        raise ValueError("artifact launch changed")
    if any(report["artifact_digest"] != artifact["digest"] for report in (state, sanitized)):
        raise ValueError("qualification evidence describes a different binary")
    if snapshot(Path(imported["repository"]) / "terminus")["digest"] != imported["source"]["digest"]:
        raise ValueError("stale Terminus source")
    if imported["resolved_config"]["model"] != campaign.immutable_model:
        raise ValueError("Terminus model differs from campaign")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(imported["resolved_config"])
    config["training"]["gradient_accumulation_steps"] = 1
    write_json(output / "config.json", config)
    schedule = balanced_schedule(policy, seed)
    plan = {"policy": policy.model_dump(mode="json"), "schedule": schedule, "seed": seed,
            "warmup": warmup, "aa": aa, "candidate": candidate_id, "config": config}
    write_json(output / "plan.json", store.envelope(plan))
    blocks = []
    lease_path = Path.home() / ".cache/terminus-autokernel/leases" / f"{campaign.gpu_uuid}.lock"
    with Lease(store.root / "controller.lock") as controller, Lease(lease_path) as lease, \
         store.activity(campaign.identity, campaign.budget,
                        seconds=2 * policy.required_pairs * timeout, kind="aa_collection" if aa else "confirmation"):
        hardware = inventory(campaign.gpu_uuid, campaign.power_limit_watts)
        runtime_digest = None
        for pair in schedule:
            for arm in pair["order"]:
                location = output / (pair["pair_id"] + "-" + arm)
                location.mkdir()
                request = {"repository": imported["repository"], "source_digest": imported["source"]["digest"],
                           "config": str(output / "config.json"), "batch": 8, "warmup": warmup,
                           "steps": policy.updates_per_block, "output": str(location), "save_state": False,
                           "artifact": artifact if arm == "candidate" and not aa else None}
                write_json(location / "request.json", request)
                with Telemetry(campaign.gpu_uuid, location / "telemetry.json"):
                    run_process([sys.executable, "-m", "autokernel.training_worker", "--request",
                                 str(location / "request.json")], cwd=location, timeout=timeout,
                                log_path=location / "worker.log", pass_fds=(controller.fd, lease.fd),
                                env=clean_env({"CUDA_VISIBLE_DEVICES": campaign.gpu_uuid}))
                current = inventory(campaign.gpu_uuid, campaign.power_limit_watts)
                if current != hardware:
                    raise ValueError("hardware cohort changed during collection")
                receipt = read_json(location / "benchmark.json")
                current_runtime = digest({"hardware": hardware, "runtime": receipt["runtime"],
                                          "judge": judge_digest()})
                if runtime_digest is not None and runtime_digest != current_runtime:
                    raise ValueError("software runtime changed during collection")
                runtime_digest = current_runtime
                if receipt["tokens_per_update"] != campaign.tokens_per_update:
                    raise ValueError("update tokens changed")
                block = TimingBlock(pair_id=pair["pair_id"], session=pair["session"], arm=arm,
                    candidate_digest=imported["source"]["digest"] if arm == "control" or aa else candidate_id,
                    runtime_digest=runtime_digest, workload_digest=digest(config),
                    start_state_digest=digest({"source": imported["source"]["digest"], "config": config, "warmup": warmup}),
                    input_digest=digest({"seed": config["training"]["seed"], "pool": 16, "config": config}),
                    tokens_per_update=campaign.tokens_per_update, seconds=receipt["step_seconds"])
                blocks.append(block.model_dump(mode="json"))
                write_json(output / "partial.json", store.envelope({"complete": False, "blocks": blocks}))
        write_json(output / "blocks.json", blocks)
        receipt = {"complete": True, "plan_digest": digest(plan), "blocks_digest": digest(blocks),
                   "candidate_digest": candidate_id, "runtime_digest": runtime_digest}
        write_json(output / "receipt.json", store.envelope(receipt))
        store.event({"type": "timing_collected", "receipt": receipt})
    return receipt
