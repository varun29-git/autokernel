import sys
from pathlib import Path

import pytest

from autokernel.analysis import ConfirmationPolicy
from autokernel.collection import balanced_schedule
from autokernel.search import command_proposal


def test_randomization_is_reproducible_and_balanced():
    policy = ConfirmationPolicy()
    schedule = balanced_schedule(policy, 741)
    assert schedule == balanced_schedule(policy, 741)
    assert len(schedule) == 8 and len({p["session"] for p in schedule}) == 4
    assert sum(p["order"][0] == "control" for p in schedule) == 4
    assert balanced_schedule(policy, 12) != schedule
    with pytest.raises(ValueError):
        balanced_schedule(ConfirmationPolicy(required_pairs=9), 1)


def test_command_generator_protocol(campaign):
    generator = Path(__file__).parents[1] / "examples/demo_generator.py"
    candidate, source = command_proposal([sys.executable, str(generator)], campaign.task, {})
    assert candidate.task_digest == campaign.task.identity
    assert candidate.parameters["chunk"] == 8192
    assert source == b'{"operation":"add"}'


def test_generator_cannot_print_fake_pass(campaign):
    with pytest.raises(ValueError):
        command_proposal([sys.executable, "-c", "print('PASS')"], campaign.task, {})


def test_relative_generator_executable_is_resolved_before_isolation(campaign, monkeypatch, tmp_path):
    link = tmp_path / "python"
    link.symlink_to(sys.executable)
    generator = Path(__file__).parents[1] / "examples/demo_generator.py"
    monkeypatch.chdir(tmp_path)
    candidate, _ = command_proposal(["./python", str(generator)], campaign.task, {})
    assert candidate.parameters["chunk"] == 8192
