from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pytest

from circcov import CirculantCovariance, ToeplitzCovariance, matern, uniform_grid


def _best_runtime(
    fn: Callable[[], object], *, warmup: int = 2, repeats: int = 7
) -> float:
    for _ in range(warmup):
        fn()

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
    return best


@pytest.mark.performance
def test_direct_matvec_is_faster_than_dense_toeplitz() -> None:
    n = 2048
    grid = uniform_grid(0.0, 1.0, n)
    kernel = matern(0.5, length_scale=0.3, variance=2.0)
    x = np.random.default_rng(0).standard_normal(n)

    toeplitz = ToeplitzCovariance(kernel, grid)
    direct = CirculantCovariance(kernel, grid, mode="direct")

    toeplitz_time = _best_runtime(lambda: toeplitz.matvec(x))
    direct_time = _best_runtime(lambda: direct.matvec(x))

    assert toeplitz_time / direct_time > 5.0


@pytest.mark.performance
def test_direct_solve_is_faster_than_dense_toeplitz() -> None:
    n = 1024
    grid = uniform_grid(0.0, 1.0, n)
    kernel = matern(0.5, length_scale=0.3, variance=2.0)
    x = np.random.default_rng(0).standard_normal(n)

    toeplitz = ToeplitzCovariance(kernel, grid)
    direct = CirculantCovariance(kernel, grid, mode="direct")

    toeplitz_time = _best_runtime(lambda: toeplitz.solve(x))
    direct_time = _best_runtime(lambda: direct.solve(x))

    assert toeplitz_time / direct_time > 5.0


@pytest.mark.performance
def test_direct_log_det_is_faster_end_to_end() -> None:
    n = 512
    grid = uniform_grid(0.0, 1.0, n)
    kernel = matern(0.5, length_scale=0.3, variance=2.0)

    toeplitz_time = _best_runtime(lambda: ToeplitzCovariance(kernel, grid).log_det())
    direct_time = _best_runtime(
        lambda: CirculantCovariance(kernel, grid, mode="direct").log_det()
    )

    assert toeplitz_time / direct_time > 5.0
