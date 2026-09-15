import pytest

from autokernel.analysis import ConfirmationPolicy, TimingBlock, calibrate, confirm, quality_noninferiority
from autokernel.physics import resource_bound, target_gap


def blocks(ratio=1., *, same_candidate=False, source="measured"):
    result = []
    for session in range(4):
        for pair in range(2):
            value = 0.7 + 0.00001 * session + 0.000002 * pair
            for arm, scale in (("control", 1), ("candidate", ratio)):
                result.append(TimingBlock(pair_id=str(pair), session=str(session), arm=arm,
                    candidate_digest="same" if same_candidate else arm, runtime_digest="runtime",
                    workload_digest="workload", start_state_digest="state", input_digest="inputs",
                    tokens_per_update=32768, seconds=[value * scale] * 3, source=source))
    return result


def test_aa_and_real_improvement():
    policy = ConfirmationPolicy(updates_per_block=3)
    calibration = calibrate(blocks(same_candidate=True), policy)
    assert calibration["status"] == "calibrated"
    result = confirm(blocks(.8), policy, calibration=calibration)
    assert result["status"] == "timing_pass"
    assert result["time_ratio_interval"][1] < .81
    assert not result["absolute_target_cleared"]


def test_no_gain_or_untrusted_simulation_cannot_pass():
    policy = ConfirmationPolicy(updates_per_block=3)
    calibration = calibrate(blocks(same_candidate=True), policy)
    assert confirm(blocks(), policy, calibration=calibration)["status"] == "inconclusive"
    assert confirm(blocks(.5), policy)["status"] == "inconclusive"
    simulation = calibrate(blocks(same_candidate=True, source="simulation"), policy)
    assert confirm(blocks(.5, source="simulation"), policy, calibration=simulation)["status"] == "inconclusive"


def test_fixed_sample_counts_and_pairing():
    policy = ConfirmationPolicy(updates_per_block=3)
    with pytest.raises(ValueError):
        confirm(blocks()[:-1], policy)
    data = blocks()
    data[1] = data[1].model_copy(update={"input_digest": "changed"})
    with pytest.raises(ValueError):
        confirm(data, policy)
    data = blocks()
    data[1] = data[1].model_copy(update={"runtime_digest": "other-gpu"})
    with pytest.raises(ValueError):
        confirm(data, policy)


def test_error_spending_is_bounded():
    assert sum(ConfirmationPolicy(round_index=i).alpha for i in range(1, 1000)) < .01


def test_quality_cannot_hide_single_language_regression():
    languages = "en hi ta te bn mr gu kn ml or pa".split()
    margins = {lang: .01 for lang in languages}
    pairs = [{"seed": seed, "language": lang, "candidate_tokens": 100, "reference_tokens": 100,
              "reference_nll": 2., "candidate_nll": 2.1 if lang == "ta" else 2.}
             for seed in range(3) for lang in languages]
    assert quality_noninferiority(pairs, margins)["status"] == "inconclusive"
    with pytest.raises(ValueError):
        quality_noninferiority(pairs[:-1], margins)
    pairs[0]["candidate_tokens"] = 99
    with pytest.raises(ValueError):
        quality_noninferiority(pairs, margins)


def test_physics_uses_overlap_and_correct_target():
    result = resource_bound(fp8_flops=1671e9, compulsory_bytes=3.9e9)
    assert result["optimistic_resource_bound_ms"] == pytest.approx(1.)
    assert target_gap(32768, .746044735, 67000)["remaining_ms"] == pytest.approx(256.970108, abs=1e-5)
