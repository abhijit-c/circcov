from __future__ import annotations

import numpy as np
import pytest

from circcov import matern, squared_exponential
from circcov.kernels import StationaryKernel


@pytest.mark.parametrize("variance", [1.0, 2.5])
@pytest.mark.parametrize("length_scale", [0.3, 1.0])
def test_kernel_positive(
    kernel: StationaryKernel, length_scale: float, variance: float
) -> None:
    """k(r) >= 0 for all r >= 0."""
    r = np.linspace(0.0, 5.0, 200)
    vals = kernel(r)
    assert np.all(vals >= 0.0), f"Negative kernel values: min={vals.min()}"


def test_kernel_zero_lag_matern05() -> None:
    k = matern(0.5, length_scale=0.4, variance=3.0)
    assert np.isclose(k(np.array([0.0]))[0], 3.0)


def test_kernel_zero_lag_matern15() -> None:
    k = matern(1.5, length_scale=0.4, variance=3.0)
    assert np.isclose(k(np.array([0.0]))[0], 3.0)


def test_kernel_zero_lag_matern25() -> None:
    k = matern(2.5, length_scale=0.4, variance=3.0)
    assert np.isclose(k(np.array([0.0]))[0], 3.0)


def test_kernel_zero_lag_se() -> None:
    k = squared_exponential(length_scale=0.4, variance=3.0)
    assert np.isclose(k(np.array([0.0]))[0], 3.0)


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_matern_decreasing(nu: float) -> None:
    """Matérn kernels are non-increasing on [0, 3*length_scale]."""
    ls = 0.5
    k = matern(nu, length_scale=ls, variance=1.0)
    r = np.linspace(0.0, 3.0 * ls, 300)
    vals = k(r)
    assert np.all(np.diff(vals) <= 1e-12), "Kernel is not monotonically decreasing"


def test_se_decreasing() -> None:
    ls = 0.5
    k = squared_exponential(length_scale=ls, variance=1.0)
    r = np.linspace(0.0, 3.0 * ls, 300)
    vals = k(r)
    assert np.all(np.diff(vals) <= 1e-12), "SE kernel is not monotonically decreasing"


def test_matern_invalid_nu() -> None:
    with pytest.raises(ValueError, match="nu"):
        matern(1.0)
