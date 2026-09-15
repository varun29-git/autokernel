from pathlib import Path

import pytest

from autokernel.build import compile_candidate
from autokernel.cli import demo_campaign
from autokernel.contracts import Budget, CampaignSpec, CandidateSpec, Case, TaskSpec
from autokernel.diagnostics import diagnose
from autokernel.release import export_research_bundle
from autokernel.terminus import dataclass_defaults


def hopper(campaign):
    task = TaskSpec(operation="fp8_rows_bf16", equations="FP8 rows: independent scale and code equations.",
                    cases=[Case(rows=7, columns=128)], reference_digest=campaign.task.reference_digest)
    return CampaignSpec(name="native", mode="hopper", task=task, gpu_execution=False,
                        source_snapshot_digest=campaign.source_snapshot_digest,
                        immutable_model={"n_layers": 30, "dim": 2048, "ffn_dim": 5376,
                                         "vocab_size": 96000, "max_seq_len": 4096})


def test_native_build_refuses_unsafe_host_fallback(campaign, tmp_path, monkeypatch):
    native = hopper(campaign)
    candidate = CandidateSpec(task_digest=native.task.identity, backend="cuda", family="row",
        source_digest="a" * 64, parameters={"block": 128}, hypothesis="Preserve FP8 rows with a compiled CUDA kernel.")
    with pytest.raises(RuntimeError, match="disabled"):
        compile_candidate(candidate, native, b"source", tmp_path)
    enabled = native.model_copy(update={"gpu_execution": True, "gpu_uuid": "GPU-test",
                                        "build_image": "test@sha256:" + "a" * 64})
    monkeypatch.setattr("autokernel.build.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="unsafe host compilation"):
        compile_candidate(candidate, enabled, b"source", tmp_path)


def test_required_sanitizers_cannot_be_faked_by_demo(store, tmp_path):
    store.freeze(demo_campaign())
    with pytest.raises(RuntimeError, match="Hopper"):
        diagnose(store, "a" * 64, output=tmp_path / "checks")


def test_importer_never_executes_foreign_config(tmp_path):
    source = tmp_path / "configuration.py"
    marker = tmp_path / "executed"
    source.write_text(f"open({str(marker)!r},'w').write('unsafe')\nclass Config:\n    layers: int = 30\n")
    assert dataclass_defaults(source, "Config") == {"layers": 30}
    assert not marker.exists()
    source.write_text("class Config:\n    layers: int = dangerous()\n")
    with pytest.raises(ValueError, match="nonliteral"):
        dataclass_defaults(source, "Config")


def test_activity_failure_is_charged(store, campaign):
    store.freeze(campaign)
    with pytest.raises(RuntimeError, match="failed"):
        with store.activity(campaign.identity, campaign.budget, seconds=5, kind="diagnostic"):
            raise RuntimeError("failed")
    state = store.db.execute("SELECT spent,reserved FROM budgets").fetchone()
    assert state[0] >= 0 and state[1] == 0
    assert store.db.execute("SELECT settled FROM reservations").fetchone()[0] == 1


def test_confirmation_reserve_cannot_be_spent_by_search(store, campaign):
    from autokernel.search import seed_candidates

    budget = Budget(max_wall_seconds=100., reserve_confirmation_seconds=80.)
    candidate = next(seed_candidates(store, campaign))
    store.enqueue(campaign.identity, candidate)
    with pytest.raises(RuntimeError, match="budget"):
        store.claim(campaign.identity, candidate.identity, budget, 30.)


def test_source_and_manifest_tampering_prevents_export(store, tmp_path):
    from autokernel.controller import run_campaign
    from autokernel.integrity import read_json, write_json
    from autokernel.search import seed_candidates

    campaign = demo_campaign()
    candidate = next(seed_candidates(store, campaign))
    run_campaign(store, campaign, [candidate])
    row = store.rows()[0]
    path = store.root / "attempts" / row["attempt"] / "artifact.json"
    artifact = read_json(path)
    artifact["metadata"]["arbitrary_new_launch"] = True
    write_json(path, artifact)
    with pytest.raises(ValueError, match="launch manifest"):
        export_research_bundle(store, candidate.identity, tmp_path / "export")


def test_hopper_templates_are_packaged():
    from autokernel import build

    root = Path(build.__file__).parent / "templates"
    assert (root / "fp8_rows.cu").is_file() and (root / "fp8_rows.py").is_file()
