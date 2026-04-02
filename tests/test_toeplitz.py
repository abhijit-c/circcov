from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from circcov import ToeplitzCovariance
from circcov.kernels import StationaryKernel

ATOL = 1e-8


def test_matvec_consistency(
    kernel: StationaryKernel,
    grid: NDArray[np.float64],
    rng: Generator,
) -> None:
    """ToeplitzCovariance.matvec(x) matches to_dense() @ x."""
    T = ToeplitzCovariance(kernel, grid)
    x = rng.standard_normal(len(grid))
    np.testing.assert_allclose(T.matvec(x), T.to_dense() @ x, atol=ATOL)


def test_matvec_consistency_2d(
    kernel2d: StationaryKernel,
    grid2d: tuple[np.ndarray, np.ndarray],
    rng: Generator,
) -> None:
    T = ToeplitzCovariance(kernel2d, grid2d)
    x = rng.standard_normal(grid2d[0].size * grid2d[1].size)
    np.testing.assert_allclose(T.matvec(x), T.to_dense() @ x, atol=ATOL)


def test_solve_consistency(
    kernel: StationaryKernel,
    grid: NDArray[np.float64],
    rng: Generator,
) -> None:
    """T.matvec(T.solve(x)) ≈ x."""
    T = ToeplitzCovariance(kernel, grid)
    x = rng.standard_normal(len(grid))
    np.testing.assert_allclose(T.matvec(T.solve(x)), x, atol=ATOL)


def test_solve_consistency_2d(
    kernel2d: StationaryKernel,
    grid2d: tuple[np.ndarray, np.ndarray],
    rng: Generator,
) -> None:
    T = ToeplitzCovariance(kernel2d, grid2d)
    size = grid2d[0].size * grid2d[1].size
    x = rng.standard_normal(size)
    np.testing.assert_allclose(T.matvec(T.solve(x)), x, atol=ATOL)


def test_log_det_consistency(
    kernel: StationaryKernel,
    grid: NDArray[np.float64],
) -> None:
    """T.log_det() matches np.linalg.slogdet(T.to_dense())."""
    T = ToeplitzCovariance(kernel, grid)
    _, ref = np.linalg.slogdet(T.to_dense())
    assert abs(T.log_det() - float(ref)) < ATOL * max(1.0, abs(float(ref)))


def test_log_det_consistency_2d(
    kernel2d: StationaryKernel,
    grid2d: tuple[np.ndarray, np.ndarray],
) -> None:
    T = ToeplitzCovariance(kernel2d, grid2d)
    _, ref = np.linalg.slogdet(T.to_dense())
    assert abs(T.log_det() - float(ref)) < ATOL * max(1.0, abs(float(ref)))


def test_sample_covariance_matches_dense(
    kernel: StationaryKernel,
    grid: NDArray[np.float64],
    rng: Generator,
) -> None:
    """Empirical covariance of samples matches the dense covariance."""
    T = ToeplitzCovariance(kernel, grid)
    samples = T.sample(8_000, rng)
    empirical = np.cov(samples)
    np.testing.assert_allclose(empirical, T.to_dense(), atol=2.5e-1)


def test_sample_covariance_matches_dense_2d(
    kernel2d: StationaryKernel,
    grid2d: tuple[np.ndarray, np.ndarray],
    rng: Generator,
) -> None:
    T = ToeplitzCovariance(kernel2d, grid2d)
    samples = T.sample(4_000, rng)
    empirical = np.cov(samples)
    np.testing.assert_allclose(empirical, T.to_dense(), atol=3.5e-1)


def test_nonuniform_grid_requires_opt_out(
    kernel: StationaryKernel,
) -> None:
    grid = np.array([0.0, 0.2, 0.5, 0.9], dtype=np.float64)

    with np.testing.assert_raises_regex(ValueError, "uniformly spaced"):
        ToeplitzCovariance(kernel, grid)

    T = ToeplitzCovariance(kernel, grid, check_grid=False)
    expected = kernel(np.abs(grid[:, None] - grid[None, :]))
    expected[np.diag_indices_from(expected)] += 1e-6
    np.testing.assert_allclose(T.to_dense(), expected, atol=ATOL)


def test_nonuniform_grid_requires_opt_out_2d(
    kernel2d: StationaryKernel,
) -> None:
    grid = (
        np.array([0.0, 0.25, 0.7], dtype=np.float64),
        np.array([0.0, 0.4, 0.8], dtype=np.float64),
    )

    with np.testing.assert_raises_regex(ValueError, "uniformly spaced"):
        ToeplitzCovariance(kernel2d, grid)

    T = ToeplitzCovariance(kernel2d, grid, check_grid=False)
    assert T.to_dense().shape == (9, 9)


def test_short_grid_raises(
    kernel: StationaryKernel,
) -> None:
    with np.testing.assert_raises_regex(ValueError, "at least two points"):
        ToeplitzCovariance(kernel, np.array([0.0], dtype=np.float64))


def test_short_grid_raises_2d(
    kernel2d: StationaryKernel,
) -> None:
    with np.testing.assert_raises_regex(ValueError, "at least two points"):
        ToeplitzCovariance(
            kernel2d,
            (np.array([0.0], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64)),
        )
