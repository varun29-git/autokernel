import numpy as np
import pytest

from autokernel.contracts import Case, NumericalPolicy
from autokernel.oracles import compare, fixture, fp8_encode, from_bf16, reference, to_bf16


def test_fp8_known_values_and_ties_to_even():
    values = np.array([0, -0., 1, -1, 448, -448, 1.0625, 1.1875, 2**-9, np.nan], dtype=np.float32)
    assert fp8_encode(values).tolist() == [0, 128, 56, 184, 126, 254, 56, 58, 1, 127]


def test_bf16_roundtrip_and_nan():
    values = np.array([1, 1 + 2**-8, 1 + 3 * 2**-8, 0, -0., np.nan, np.inf], dtype=np.float32)
    restored = from_bf16(to_bf16(values))
    assert restored[:3].tolist() == [1, 1, 1.015625]
    assert np.signbit(restored[4]) and np.isnan(restored[5]) and np.isposinf(restored[6])


@pytest.mark.parametrize("stress", ["normal", "zero", "tiny", "outlier", "rounding", "nonfinite"])
def test_quantization_with_strides_and_tails(stress):
    case = Case(rows=7, columns=256, column_stride=2, row_padding=8, stress=stress, epsilon=0.)
    inputs = fixture(case, "fp8_rows_bf16", 12)
    before = inputs["input"].copy()
    expected = reference(case, "fp8_rows_bf16", inputs)
    assert expected["codes"].shape == (7, 256) and expected["scales"].shape == (7, 2)
    assert np.array_equal(inputs["input"], before)
    compare(expected, expected, NumericalPolicy())
    wrong = {key: value.copy() for key, value in expected.items()}
    wrong["codes"][-1, -1] ^= 1
    with pytest.raises(AssertionError):
        compare(wrong, expected, NumericalPolicy())


def test_zero_and_tiny_gradients_are_not_silently_erased():
    zero = Case(rows=1, columns=128, stress="zero", epsilon=0.)
    actual = reference(zero, "fp8_rows_bf16", fixture(zero, "fp8_rows_bf16", 1))
    assert np.all(actual["codes"] == 0) and np.all(actual["scales"] == 1)
    tiny = Case(rows=1, columns=128, stress="tiny", epsilon=0.)
    actual = reference(tiny, "fp8_rows_bf16", fixture(tiny, "fp8_rows_bf16", 1))
    assert np.count_nonzero(actual["codes"] & 127) > 100
    assert float(actual["scales"].max()) < 1e-12


def test_nonfinite_and_shape_mismatch_fail():
    expected = {"output": np.array([0., np.nan, np.inf], dtype=np.float32)}
    for actual in ({"output": np.zeros(3, dtype=np.float32)},
                   {"output": np.zeros(2, dtype=np.float32)},
                   {"output": np.array([0., np.nan, np.inf], dtype=np.float64)}):
        with pytest.raises(AssertionError):
            compare(actual, expected, NumericalPolicy())
