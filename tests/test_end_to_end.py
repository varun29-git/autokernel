import json

import pytest

from autokernel.build import compile_candidate
from autokernel.cli import demo_campaign
from autokernel.controller import run_campaign
from autokernel.release import export_research_bundle
from autokernel.search import rank_candidates, seed_candidates


def test_complete_search_rejects_wrong_outputs_and_mutation(store, tmp_path):
    campaign = demo_campaign()
    seeds = list(seed_candidates(store, campaign, include_wrong=True))
    # One implementation from each adversarial family and two legitimate schedules.
    selected = [seeds[i] for i in (0, 2, 3, 6, 9)]
    report = run_campaign(store, campaign, selected)
    assert sum(r["state"] == "numerical_pass" for r in report["candidates"]) == 2
    assert sum(r["state"] == "rejected" for r in report["candidates"]) == 3
    assert report["release_status"] == "not_qualified"
    before = [r["attempt"] for r in store.rows()]
    run_campaign(store, campaign, selected)
    assert before == [r["attempt"] for r in store.rows()]
    store.verify_events()
    bundle = export_research_bundle(store, report["best_exploratory_candidate"], tmp_path / "bundle")
    assert not bundle["production_qualified"] and bundle["scope"] == "cpu_demo"
    failed = next(r for r in report["candidates"] if r["state"] == "rejected")
    with pytest.raises(ValueError):
        export_research_bundle(store, failed["candidate"], tmp_path / "wrong")


def test_arbitrary_demo_python_is_never_executed(store, campaign, tmp_path):
    candidate = next(seed_candidates(store, campaign))
    with pytest.raises((ValueError, json.JSONDecodeError)):
        compile_candidate(candidate, campaign, b"import os; os.system('touch unsafe')", tmp_path)
    assert not (tmp_path / "unsafe").exists()


def test_search_preserves_family_diversity(store, campaign):
    candidates = list(seed_candidates(store, campaign, include_wrong=True))
    ranked = rank_candidates(candidates, beam_families=4)
    assert len({c.family for c in ranked[:4]}) == 4
    assert {c.identity for c in ranked} == {c.identity for c in candidates}
