from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import oracles
from .analysis import ConfirmationPolicy, TimingBlock, calibrate, confirm, quality_noninferiority
from .contracts import Budget, CampaignSpec, CandidateSpec, Case, TaskSpec, schemas
from .controller import campaign_report, run_campaign
from .evaluator import judge_digest
from .integrity import digest, file_digest, read_json, write_json
from .physics import target_gap
from .process import Lease
from .release import export_research_bundle
from .search import command_proposal, rank_candidates, seed_candidates
from .store import Store
from .terminus import historical_receipt, hopper_campaign, import_terminus


def demo_campaign():
    task = TaskSpec(operation="vector_add_f32", equations="Elementwise FP32 addition; immutable strided inputs.",
                    reference_digest=file_digest(oracles.__file__),
                    cases=[Case(rows=8, columns=4096), Case(rows=3, columns=257, column_stride=2, row_padding=8),
                           Case(rows=2, columns=129, stress="tiny"), Case(rows=1, columns=128, stress="nonfinite")])
    return CampaignSpec(name="local-demo", source_snapshot_digest=judge_digest(), task=task,
                        budget=Budget(max_candidates=16, worker_timeout_seconds=10.0), timing_repetitions=5)


def generate_one(store, campaign, argv):
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise ValueError("explicit generator argv required")
    with Lease(store.root / "controller.lock"):
        calls = store.db.execute("SELECT payload FROM events WHERE payload LIKE '%generator_reserved%'").fetchall()
        if len(calls) >= campaign.budget.max_generator_calls:
            raise RuntimeError("generator call budget exhausted")
        with store.transaction():
            store.event({"type": "generator_reserved", "campaign": campaign.identity})
        with store.activity(campaign.identity, campaign.budget, seconds=120., kind="generation"):
            observations = campaign_report(store)
            candidate, source = command_proposal(argv, campaign.task, observations)
        store.event({"type": "agent_proposal", "candidate": candidate.identity,
                     "source": candidate.source_digest, "observation_digest": digest(observations)})
        if store.put(source) != candidate.source_digest:
            raise ValueError("generator source changed")
        return candidate


def parser():
    root = argparse.ArgumentParser(description="Reproducible kernel search with independent correctness gates")
    sub = root.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run a real CPU search; intentionally wrong kernels must fail")
    demo.add_argument("--store", required=True)
    demo.add_argument("--include-wrong", action="store_true")
    demo_plan = sub.add_parser("demo-plan", help="Write a CPU-only campaign with a bounded generator allowance")
    demo_plan.add_argument("--output", required=True)
    demo_plan.add_argument("--generator-calls", type=int, default=0)
    schema = sub.add_parser("schemas", help="Export versioned JSON schemas")
    schema.add_argument("--output", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("campaign")
    importer = sub.add_parser("import-terminus", help="Freeze config/source without importing or executing training")
    importer.add_argument("--repo", required=True)
    importer.add_argument("--config", required=True)
    importer.add_argument("--output", required=True)
    hopper = sub.add_parser("hopper-plan", help="Generate a disabled Hopper campaign; no GPU activity")
    hopper.add_argument("--imported", required=True)
    hopper.add_argument("--output", required=True)
    hopper.add_argument("--gpu-uuid")
    hopper.add_argument("--build-image")
    hopper.add_argument("--enable-execution", action="store_true")
    hopper.add_argument("--generator-calls", type=int, default=12)
    hopper.add_argument("--seed-candidates", type=int, default=4)
    run = sub.add_parser("run", help="Run/resume a frozen bounded campaign on this machine")
    run.add_argument("--campaign", required=True)
    run.add_argument("--store", required=True)
    research = sub.add_parser("research", help="Alternate bounded generator proposals and independent evaluation")
    research.add_argument("--campaign", required=True)
    research.add_argument("--store", required=True)
    research.add_argument("generator", nargs=argparse.REMAINDER)
    collect = sub.add_parser("collect-pairs", help="Collect a frozen randomized native training comparison")
    collect.add_argument("--store", required=True)
    collect.add_argument("--candidate", required=True)
    collect.add_argument("--imported", required=True)
    collect.add_argument("--policy", required=True)
    collect.add_argument("--integration-report", required=True)
    collect.add_argument("--sanitizer-report", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--aa", action="store_true")
    collect.add_argument("--seed", type=int, default=741)
    collect.add_argument("--warmup", type=int, default=32)
    collect.add_argument("--timeout", type=int, default=600)
    for command in ("sanitize", "profile"):
        p = sub.add_parser(command)
        p.add_argument("--store", required=True)
        p.add_argument("--candidate", required=True)
        p.add_argument("--output", required=True)
    proposal = sub.add_parser("propose", help="Ask an explicitly configured generator for one candidate")
    proposal.add_argument("--store", required=True)
    proposal.add_argument("generator", nargs=argparse.REMAINDER)
    submit = sub.add_parser("submit", help="Validate and enqueue a manually generated candidate")
    submit.add_argument("--store", required=True)
    submit.add_argument("--candidate", required=True)
    submit.add_argument("--source", required=True)
    for command in ("report", "verify-store"):
        p = sub.add_parser(command)
        p.add_argument("--store", required=True)
    historical = sub.add_parser("inspect-historical")
    historical.add_argument("receipt")
    for command in ("calibrate", "confirm"):
        p = sub.add_parser(command)
        p.add_argument("--blocks", required=True)
        p.add_argument("--policy", required=True)
        p.add_argument("--output", required=True)
        if command == "confirm":
            p.add_argument("--calibration", required=True)
    quality = sub.add_parser("quality-analysis")
    quality.add_argument("--pairs", required=True)
    quality.add_argument("--margins", required=True)
    quality.add_argument("--output", required=True)
    export = sub.add_parser("export", help="Export a research bundle; cannot certify production")
    export.add_argument("--store", required=True)
    export.add_argument("--candidate", required=True)
    export.add_argument("--output", required=True)
    train = sub.add_parser("training-pair", help="Native complete-update/state comparison; never starts cloud VMs")
    train.add_argument("--store", required=True)
    train.add_argument("--candidate", required=True)
    train.add_argument("--imported", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--steps", type=int, default=8)
    train.add_argument("--warmup", type=int, default=5)
    train.add_argument("--timeout", type=int, default=1800)
    train.add_argument("--no-state", action="store_true")
    gap = sub.add_parser("gap")
    gap.add_argument("--tokens", type=int, default=32768)
    gap.add_argument("--seconds", type=float, required=True)
    gap.add_argument("--target", type=float, default=67000)
    return root


def dispatch(args):
    if args.command == "demo-plan":
        value = demo_campaign().model_dump(mode="json")
        value["budget"]["max_generator_calls"] = args.generator_calls
        campaign = CampaignSpec.model_validate(value)
        write_json(args.output, campaign.model_dump(mode="json"))
        return {"campaign": args.output, "gpu_execution": False}
    if args.command == "schemas":
        path = Path(args.output)
        path.mkdir(parents=True, exist_ok=True)
        for name, schema in schemas().items():
            write_json(path / f"{name}.schema.json", schema)
        return {"schemas": sorted(schemas())}
    if args.command == "validate":
        campaign = CampaignSpec.model_validate(read_json(args.campaign))
        return {"campaign_digest": campaign.identity, "mode": campaign.mode, "gpu_execution": campaign.gpu_execution}
    if args.command == "import-terminus":
        return import_terminus(args.repo, args.config, args.output)
    if args.command == "hopper-plan":
        campaign = hopper_campaign(read_json(args.imported), gpu_uuid=args.gpu_uuid,
                                   build_image=args.build_image, enable=args.enable_execution,
                                   generator_calls=args.generator_calls, seed_candidates=args.seed_candidates)
        write_json(args.output, campaign.model_dump(mode="json"))
        return {"campaign": args.output, "gpu_execution": campaign.gpu_execution}
    if args.command == "inspect-historical":
        return historical_receipt(args.receipt)
    if args.command in {"calibrate", "confirm"}:
        blocks = [TimingBlock.model_validate(row) for row in read_json(args.blocks)]
        policy = ConfirmationPolicy.model_validate(read_json(args.policy))
        report = (calibrate(blocks, policy) if args.command == "calibrate" else
                  confirm(blocks, policy, calibration=read_json(args.calibration)))
        write_json(args.output, report)
        return report
    if args.command == "quality-analysis":
        report = quality_noninferiority(read_json(args.pairs), read_json(args.margins))
        write_json(args.output, report)
        return report
    if args.command == "gap":
        return target_gap(args.tokens, args.seconds, args.target)
    store = Store(args.store)
    try:
        if args.command == "demo":
            campaign = demo_campaign()
            return run_campaign(store, campaign, list(seed_candidates(store, campaign, include_wrong=args.include_wrong)))
        if args.command == "run":
            campaign = CampaignSpec.model_validate(read_json(args.campaign))
            return run_campaign(store, campaign, list(seed_candidates(store, campaign)))
        if args.command == "research":
            campaign = CampaignSpec.model_validate(read_json(args.campaign))
            seeds = rank_candidates(list(seed_candidates(store, campaign)), beam_families=campaign.beam_families)
            run_campaign(store, campaign, seeds[:campaign.initial_seed_candidates])
            for _ in range(campaign.budget.max_generator_calls):
                try:
                    candidate = generate_one(store, campaign, args.generator)
                except RuntimeError as error:
                    store.event({"type": "generation_stopped", "reason": str(error)})
                    break
                run_campaign(store, campaign, [candidate])
            return campaign_report(store)
        if args.command == "report":
            return campaign_report(store)
        if args.command == "verify-store":
            store.verify_events()
            for row in store.rows():
                if row["receipt"]:
                    store.receipt(row["receipt"])
            store.verify(read_json(store.root / "campaign.json"))
            return {"integrity": "passed", "candidates": len(store.rows())}
        if args.command in {"submit", "propose"}:
            campaign = CampaignSpec.model_validate(store.verify(read_json(store.root / "campaign.json")))
            if args.command == "propose":
                candidate = generate_one(store, campaign, args.generator)
                source = store.get(candidate.source_digest)
            else:
                candidate = CandidateSpec.model_validate(read_json(args.candidate))
                source = Path(args.source).read_bytes()
            if candidate.task_digest != campaign.task.identity or store.put(source) != candidate.source_digest:
                raise ValueError("candidate/source/task mismatch")
            enqueued = store.enqueue(campaign.identity, candidate)
            return {"candidate": candidate.identity, "enqueued": enqueued}
        if args.command == "export":
            return export_research_bundle(store, args.candidate, args.output)
        if args.command in {"sanitize", "profile"}:
            from .diagnostics import diagnose

            return diagnose(store, args.candidate, output=args.output,
                            kind="sanitizers" if args.command == "sanitize" else "profile")
        if args.command == "training-pair":
            from .training import training_pair

            return training_pair(store, args.candidate, read_json(args.imported), output=args.output,
                                 batch=args.batch, warmup=args.warmup, steps=args.steps, timeout=args.timeout,
                                 save_state=not args.no_state)
        if args.command == "collect-pairs":
            from .collection import collect_training

            return collect_training(store, args.candidate, read_json(args.imported),
                ConfirmationPolicy.model_validate(read_json(args.policy)), output=args.output,
                integration_report=args.integration_report, sanitizer_report=args.sanitizer_report,
                aa=args.aa, seed=args.seed, warmup=args.warmup, timeout=args.timeout)
    finally:
        store.close()


def main():
    args = parser().parse_args()
    try:
        result = dispatch(args)
    except Exception as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}), file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
