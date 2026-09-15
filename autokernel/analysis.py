"""Predeclared, session-level paired inference; exploratory timing is never a certificate."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from pydantic import Field, FiniteFloat, model_validator
from scipy.stats import t

from .contracts import Contract
from .integrity import digest


class TimingBlock(Contract):
    pair_id: str
    session: str
    arm: str
    candidate_digest: str
    runtime_digest: str
    workload_digest: str
    start_state_digest: str
    input_digest: str
    tokens_per_update: int = Field(gt=0)
    seconds: list[FiniteFloat] = Field(min_length=3)
    source: str = "measured"

    @model_validator(mode="after")
    def valid(self):
        if self.arm not in {"control", "candidate"} or any(value <= 0 for value in self.seconds):
            raise ValueError("invalid timing block")
        if self.source not in {"measured", "simulation", "historical"}:
            raise ValueError("unknown timing provenance")
        return self


class ConfirmationPolicy(Contract):
    round_index: int = Field(default=1, ge=1)
    shortlist_size: int = Field(default=1, ge=1)
    campaign_alpha: FiniteFloat = Field(default=0.01, gt=0, lt=0.1)
    minimum_speedup: FiniteFloat = Field(default=0.01, ge=0)
    required_pairs: int = Field(default=8, ge=8)
    required_sessions: int = Field(default=4, ge=4)
    updates_per_block: int = Field(default=128, ge=3)
    target_tokens_per_second: FiniteFloat = Field(default=67000.0, gt=0)

    @property
    def alpha(self):
        return self.campaign_alpha / (self.round_index * (self.round_index + 1) * self.shortlist_size)


def paired_data(blocks, policy):
    if len(blocks) != 2 * policy.required_pairs:
        raise ValueError("sample count differs from frozen protocol; no optional stopping")
    pairs = defaultdict(dict)
    cohort = None
    arm_ids = defaultdict(set)
    for block in blocks:
        if len(block.seconds) != policy.updates_per_block:
            raise ValueError("update count differs from frozen protocol")
        key = (block.session, block.pair_id)
        if block.arm in pairs[key]:
            raise ValueError("duplicate timing arm")
        pairs[key][block.arm] = block
        identity = (block.runtime_digest, block.workload_digest, block.tokens_per_update, block.source)
        if cohort is not None and identity != cohort:
            raise ValueError("mixed runtime, workload, tokens or provenance")
        cohort = identity
        arm_ids[block.arm].add(block.candidate_digest)
    if any(len(values) != 1 for values in arm_ids.values()):
        raise ValueError("candidate changed during confirmation")
    sessions = defaultdict(list)
    for (session, _), pair in pairs.items():
        if set(pair) != {"control", "candidate"}:
            raise ValueError("unpaired timing block")
        a, b = pair["control"], pair["candidate"]
        if a.start_state_digest != b.start_state_digest or a.input_digest != b.input_digest:
            raise ValueError("paired initial state or inputs differ")
        sessions[session].append((sum(a.seconds), sum(b.seconds)))
    if len(sessions) != policy.required_sessions:
        raise ValueError("session count differs from frozen protocol")
    if len({len(values) for values in sessions.values()}) != 1:
        raise ValueError("balanced independent sessions are required")
    # The independent units are sessions, not individual optimizer steps.
    means = np.array([np.mean(values, axis=0) for _, values in sorted(sessions.items())])
    return means, cohort, arm_ids


def confirm(blocks, policy, *, calibration=None):
    means, cohort, arm_ids = paired_data(blocks, policy)
    n = len(means)
    x, y = means.mean(axis=0)
    covariance = np.cov(means.T, ddof=1) / n
    critical = float(t.ppf(1 - policy.alpha / 2, n - 1))
    vx, vy, cv = covariance[0, 0], covariance[1, 1], covariance[0, 1]
    # Fieller confidence set for E[candidate seconds] / E[control seconds].
    a = x * x - critical * critical * vx
    b = -2 * (x * y - critical * critical * cv)
    c = y * y - critical * critical * vy
    discriminant = b * b - 4 * a * c
    if discriminant < 0 and abs(discriminant) <= 16 * np.finfo(float).eps * (b * b + abs(4 * a * c)):
        discriminant = 0.0
    bounds = None
    if a > 0 and discriminant >= 0:
        bounds = [float((-b - np.sqrt(discriminant)) / (2 * a)),
                  float((-b + np.sqrt(discriminant)) / (2 * a))]
    upper_candidate = y + critical * np.sqrt(vy)
    tokens = cohort[2] * policy.updates_per_block
    calibrated = bool(calibration and calibration.get("status") == "calibrated"
                      and calibration.get("policy_digest") == policy.identity
                      and calibration.get("runtime_digest") == cohort[0]
                      and calibration.get("workload_digest") == cohort[1]
                      and calibration.get("source") == cohort[3])
    statistical_pass = bounds is not None and bounds[1] < 1 / (1 + policy.minimum_speedup)
    source = cohort[3]
    return {
        "scope": "session-level paired timing analysis; not a numerical or quality certificate",
        "status": "timing_pass" if statistical_pass and calibrated and source == "measured" else "inconclusive",
        "statistical_threshold_cleared": bool(statistical_pass),
        "calibrated": calibrated, "source": source, "policy_digest": policy.identity,
        "runtime_digest": cohort[0], "workload_digest": cohort[1],
        "control_digest": next(iter(arm_ids["control"])),
        "candidate_digest": next(iter(arm_ids["candidate"])),
        "alpha": policy.alpha, "independent_sessions": n,
        "time_ratio": float(y / x), "time_ratio_interval": bounds,
        "control_tokens_per_second": float(tokens / x), "candidate_tokens_per_second": float(tokens / y),
        "candidate_tokens_per_second_lower_bound": float(tokens / upper_candidate),
        "absolute_target_cleared": bool(tokens / upper_candidate >= policy.target_tokens_per_second),
        "assumptions": "Balanced independent sessions; approximately bivariate-normal session means. "
                       "A/A calibration screens obvious bias; it does not prove tail coverage.",
    }


def calibrate(blocks, policy):
    means, cohort, ids = paired_data(blocks, policy)
    if ids["control"] != ids["candidate"]:
        raise ValueError("A/A calibration requires the identical candidate in both arms")
    result = confirm(blocks, policy)
    bounds = result["time_ratio_interval"]
    unbiased = bounds is not None and bounds[0] <= 1 <= bounds[1]
    # The interval must also resolve the practical threshold, not simply be extremely wide.
    precise = bounds is not None and (bounds[1] - bounds[0]) < 2 * policy.minimum_speedup
    return {"status": "calibrated" if unbiased and precise else "inconclusive",
            "policy_digest": policy.identity, "runtime_digest": cohort[0], "workload_digest": cohort[1],
            "source": cohort[3], "time_ratio_interval": bounds,
            "input_digest": digest([b.model_dump(mode="json") for b in blocks]),
            "scope": "A/A bias and resolution check; use a separate fixed confirmation sample"}


def quality_noninferiority(pairs, margins, *, alpha=0.01):
    """Pairs contain seed-level held-out NLL means after equal-token paired training.

    Seed is the independent unit. Requires every language for every paired seed;
    caller must attest data/checkpoint identities before these statistics are evidence.
    """
    languages = {"en", "hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "or", "pa"}
    if set(margins) != languages or not 0 < alpha < 0.1:
        raise ValueError("explicit margins for all eleven languages and valid alpha required")
    if any(not np.isfinite(v) or v < 0 for v in margins.values()):
        raise ValueError("invalid noninferiority margin")
    seeds = defaultdict(dict)
    for row in pairs:
        if row["language"] not in languages or row["language"] in seeds[row["seed"]]:
            raise ValueError("unknown or duplicate seed/language")
        if row["candidate_tokens"] != row["reference_tokens"] or row["candidate_tokens"] <= 0:
            raise ValueError("quality comparison must use equal positive token counts")
        if not np.isfinite([row["candidate_nll"], row["reference_nll"]]).all():
            raise ValueError("nonfinite quality measurement")
        seeds[row["seed"]][row["language"]] = row["candidate_nll"] - row["reference_nll"]
    if len(seeds) < 3 or any(set(row) != languages for row in seeds.values()):
        raise ValueError("at least three complete paired seeds are required")
    report = {}
    for language in sorted(languages):
        values = np.array([row[language] for row in seeds.values()], dtype=float)
        upper = float(values.mean() + t.ppf(1 - alpha / len(languages), len(values) - 1)
                      * values.std(ddof=1) / np.sqrt(len(values)))
        report[language] = {"mean_nll_difference": float(values.mean()), "upper_bound": upper,
                            "margin": margins[language], "passed": bool(upper <= margins[language])}
    return {"status": "statistical_pass" if all(row["passed"] for row in report.values()) else "inconclusive",
            "languages": report, "scope": "Statistics only; data provenance and pilot sensitivity require qualification"}
