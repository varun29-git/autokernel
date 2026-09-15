import json

import pytest
from pydantic import ValidationError

from autokernel.contracts import CampaignSpec, CandidateSpec, Case, TaskSpec, schemas
from autokernel.integrity import read_json, snapshot


def test_unknown_fields_and_nonfinite_are_rejected(campaign):
    value = campaign.model_dump(mode="json")
    value["skip_correctness"] = True
    with pytest.raises(ValidationError):
        CampaignSpec.model_validate(value)
    with pytest.raises(ValidationError):
        Case(rows=1, columns=128, epsilon=float("nan"))
    with pytest.raises(ValidationError):
        Case(rows=True, columns=128)


def test_hopper_cannot_be_enabled_without_identity(campaign):
    payload = campaign.model_dump(mode="json")
    payload.update(mode="hopper", gpu_execution=True)
    payload["task"]["operation"] = "fp8_rows_bf16"
    with pytest.raises(ValidationError):
        CampaignSpec.model_validate(payload)


def test_launch_manifest_rejects_arbitrary_host_parameters():
    with pytest.raises(ValidationError):
        CandidateSpec(task_digest="a" * 64, backend="cuda", family="row", source_digest="b" * 64,
                      parameters={"block": 128, "skip_sync": 1}, hypothesis="Attempt an unauthorized parameter")


def test_fp8_policy_cannot_relax_exactness(campaign):
    payload = campaign.task.model_dump(mode="json")
    payload.update(operation="fp8_rows_bf16", cases=[Case(rows=1, columns=128).model_dump()])
    payload["numerical"]["exact"] = False
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(payload)


def test_duplicate_json_and_nan(tmp_path):
    path = tmp_path / "bad.json"
    for text in ('{"x":1,"x":2}', '{"x":NaN}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            read_json(path)


def test_recursive_snapshot_binds_headers_and_ignores_outputs(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/kernel.cuh").write_text("one")
    first = snapshot(tmp_path)
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/output.json").write_text("{}")
    assert snapshot(tmp_path) == first
    (tmp_path / "nested/kernel.cuh").write_text("two")
    assert snapshot(tmp_path)["digest"] != first["digest"]
    (tmp_path / "linked.py").symlink_to(tmp_path / "nested/kernel.cuh")
    with pytest.raises(ValueError):
        snapshot(tmp_path)


def test_schema_export_and_roundtrip(campaign):
    assert CampaignSpec.model_validate(json.loads(campaign.model_dump_json())).identity == campaign.identity
    assert schemas()["CampaignSpec"]["additionalProperties"] is False
