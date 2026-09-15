import os
import subprocess
import sys

import pytest

from autokernel.contracts import EvaluationReceipt, Gate
from autokernel.process import Lease, clean_env, run_process
from autokernel.search import seed_candidates


def test_receipts_tamper_and_event_chain(store, campaign):
    store.freeze(campaign)
    candidate = next(seed_candidates(store, campaign))
    assert store.enqueue(campaign.identity, candidate)
    assert not store.enqueue(campaign.identity, candidate)
    attempt = store.claim(campaign.identity, candidate.identity, campaign.budget, 10.)
    receipt = EvaluationReceipt(campaign_digest=campaign.identity, candidate_digest=candidate.identity,
        task_digest=campaign.task.identity, judge_digest=campaign.source_snapshot_digest, attempt_id=attempt,
        scope="cpu_demo", gates={"numerical": Gate(status="passed")})
    identity = store.finish(candidate.identity, attempt, receipt, 1.)
    assert store.receipt(identity)["scope"] == "cpu_demo"
    store.verify_events()
    with pytest.raises(RuntimeError):
        store.finish(candidate.identity, attempt, receipt, 1.)
    envelope = store.envelope({"passed": True})
    envelope["payload"]["passed"] = False
    with pytest.raises(ValueError):
        store.verify(envelope)
    store.db.execute("UPDATE events SET payload='{}' WHERE seq=1")
    with pytest.raises(ValueError):
        store.verify_events()


def test_recovery_charges_budget_and_cannot_promote(store, campaign):
    store.freeze(campaign)
    candidate = next(seed_candidates(store, campaign))
    store.enqueue(campaign.identity, candidate)
    store.claim(campaign.identity, candidate.identity, campaign.budget, 20.)
    assert store.recover() == 1
    assert store.recover() == 0
    row = store.rows()[0]
    assert row["state"] == "interrupted" and row["receipt"] is None
    assert store.db.execute("SELECT spent,reserved FROM budgets").fetchone()[0] == 20.


def test_budget_and_campaign_immutability(store, campaign):
    store.freeze(campaign)
    candidate = next(seed_candidates(store, campaign))
    store.enqueue(campaign.identity, candidate)
    with pytest.raises(ValueError):
        store.claim(campaign.identity, candidate.identity, campaign.budget, 10000.)
    modified = campaign.model_copy(update={"name": "different"})
    with pytest.raises(ValueError):
        store.freeze(modified)


def test_content_addressed_tamper_and_traversal(store):
    identity = store.put(b"real")
    (store.objects / identity).write_bytes(b"tampered")
    with pytest.raises(ValueError):
        store.get(identity)
    with pytest.raises(ValueError):
        store.get("../supervisor.key")


def test_live_lease_cannot_be_stolen(tmp_path):
    with Lease(tmp_path / "device.lock"):
        with pytest.raises(RuntimeError):
            with Lease(tmp_path / "device.lock"):
                pass
    with Lease(tmp_path / "device.lock"):
        pass


def test_process_timeout_and_nonzero_exit(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        run_process([sys.executable, "-c", "import time; time.sleep(10)"], cwd=tmp_path, timeout=.2,
                    log_path=tmp_path / "timeout.log")
    with pytest.raises(RuntimeError):
        run_process([sys.executable, "-c", "print('PASS'); raise SystemExit(2)"], cwd=tmp_path, timeout=2,
                    log_path=tmp_path / "fakepass.log")


def test_worker_environment_does_not_inherit_credentials(monkeypatch):
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "test-value")
    monkeypatch.setenv("OPENAI_API_KEY", "test-value")
    assert "AZURE_CLIENT_SECRET" not in clean_env() and "OPENAI_API_KEY" not in clean_env()
    assert os.environ["OPENAI_API_KEY"] == "test-value"
