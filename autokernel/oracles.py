"""Independent NumPy equations; never import candidate or Terminus implementation helpers."""
from __future__ import annotations

import numpy as np

from .contracts import Case


def to_bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    result = (rounded >> 16).astype(np.uint16)
    # Preserve NaNs rather than rounding certain payloads into infinities.
    return np.where(np.isnan(values), np.uint16(0x7FC0), result).astype(np.uint16)


def from_bf16(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def fp8_encode(values):
    positive_codes = np.arange(127, dtype=np.uint8)
    exponents = (positive_codes.astype(np.int32) >> 3)
    mantissa = (positive_codes & 7).astype(np.float32)
    table = np.where(exponents == 0, mantissa * np.float32(2**-9),
                     (1 + mantissa / 8) * np.exp2(exponents - 7)).astype(np.float32)
    values = np.asarray(values, dtype=np.float32)
    finite = np.nan_to_num(np.abs(values), nan=0, posinf=448, neginf=448).clip(0, 448)
    hi = np.searchsorted(table, finite).clip(0, 126)
    lo = (hi - 1).clip(0, 126)
    delta_hi, delta_lo = table[hi] - finite, finite - table[lo]
    choose_hi = (delta_hi < delta_lo) | ((delta_hi == delta_lo) & ((hi & 1) == 0))
    codes = np.where(choose_hi, hi, lo).astype(np.uint8)
    codes |= (np.signbit(values).astype(np.uint8) << 7)
    return np.where(np.isnan(values), np.uint8(127), codes).astype(np.uint8)


def fixture(case: Case, operation, seed):
    rng = np.random.default_rng(seed)
    shape = (case.rows, case.row_stride)
    values = rng.standard_normal(shape).astype(np.float32)
    if case.stress == "zero":
        values.fill(0)
    elif case.stress == "tiny":
        values *= np.float32(1e-12)
    elif case.stress == "outlier":
        values *= np.logspace(-6, 6, case.rows, dtype=np.float32)[:, None]
    elif case.stress == "nonfinite":
        values.flat[0] = np.nan
        if values.size > 2:
            values.flat[case.column_stride] = np.inf
    elif case.stress == "rounding":
        values = rng.integers(-512, 513, size=shape).astype(np.float32) / 256
    if operation == "fp8_rows_bf16":
        return {"input": to_bf16(values)}
    return {"input": values, "rhs": rng.standard_normal(shape).astype(np.float32)}


def logical(array, case):
    return array[:, :case.columns * case.column_stride:case.column_stride]


def reference(case, operation, inputs):
    if operation == "vector_add_f32":
        return {"output": logical(inputs["input"], case) + logical(inputs["rhs"], case)}
    values = logical(from_bf16(inputs["input"]), case).reshape(case.rows, -1, 128)
    amplitude = np.maximum(np.abs(values).max(axis=-1), np.float32(case.epsilon))
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        scales = np.where(amplitude > 0,
                          np.maximum(amplitude * np.float32(1 / 448), np.finfo(np.float32).tiny),
                          np.where(amplitude == 0, np.float32(1), amplitude)).astype(np.float32)
        quotient = np.clip(values / scales[..., None], -448, 448)
    return {"codes": fp8_encode(quotient).reshape(case.rows, case.columns), "scales": scales}


def compare(actual, expected, policy):
    if set(actual) != set(expected):
        raise AssertionError("wrong output set")
    metrics = {}
    for name, ref in expected.items():
        got = np.asarray(actual[name])
        if got.shape != ref.shape or got.dtype != ref.dtype:
            raise AssertionError(f"{name}: shape or dtype changed")
        if np.issubdtype(ref.dtype, np.integer):
            # Both signs of the FP8 NaN code carry the same semantic value.
            same = got == ref
            if name == "codes" and policy.matching_nonfinite:
                same |= ((got & 127) == 127) & ((ref & 127) == 127)
            if not same.all():
                raise AssertionError(f"{name}: {int((~same).sum())} unequal codes")
            metrics[name] = {"exact": True}
            continue
        finite = np.isfinite(ref)
        if not policy.matching_nonfinite and not finite.all():
            raise AssertionError(f"{name}: nonfinite output forbidden by policy")
        if not np.array_equal(np.isnan(got), np.isnan(ref)):
            raise AssertionError(f"{name}: NaN propagation mismatch")
        if not np.array_equal(np.isposinf(got), np.isposinf(ref)) or not np.array_equal(np.isneginf(got), np.isneginf(ref)):
            raise AssertionError(f"{name}: infinity propagation mismatch")
        difference = got[finite].astype(np.float64) - ref[finite].astype(np.float64)
        absolute = float(np.abs(difference).max(initial=0))
        norm = float(np.linalg.norm(ref[finite].astype(np.float64)))
        relative = float(np.linalg.norm(difference) / norm) if norm else (0.0 if absolute == 0 else None)
        if policy.exact and absolute != 0:
            raise AssertionError(f"{name}: exact equality failed; max abs {absolute}")
        if not policy.exact and (absolute > policy.max_absolute or relative is None or relative > policy.max_relative_l2):
            raise AssertionError(f"{name}: numerical bounds exceeded")
        metrics[name] = {"max_absolute": absolute, "relative_l2": relative, "exact": absolute == 0}
    return metrics
