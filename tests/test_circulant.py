from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator
from numpy.typing import NDArray

from circcov import CirculantCovariance, ToeplitzCovariance
from circcov.kernels import StationaryKernel

ATOL = 1e-8


# ---------------------------------------------------------------------------
# embed mode
# ---------------------------------------------------------------------------


def test_matvec_consistency_embed(
    embed_kernel: StationaryKernel,
    grid: NDArray[np.float64],
    rng: Generator,
) -> None:
    """CirculantCovariance(embed).matvec(x) matches to_dense() @ x."""
    C = CirculantCovariance(embed_kernel, grid, mode="embed")
    x = rng.standard_normal(len(grid))
    np.testing.assert_allclose(C.matvec(x), C.to_dense() @ x, atol=ATOL)


def test_matvec_consistency_embed_2d(
    embed_kernel2d: StationaryKernel,
    grid2d: tuple[np.ndarray, np.ndarray],
    rng: Generator,
) -> None:
    C = CirculantCovariance(embed_kernel2d, grid2d, mode="embed")
    x = rng.standard_normal(grid2d[0].size * grid2d[1].size)
    np.testing.assert_allclose(C.matvec(x), C.to_dense() @ x, atol=ATOL)


def test_solve_consistency_embed(
    embed_kernel: StationaryKernel,
    grid: NDArray[np.float64],
    rng: Generator,
) -> None:
    """C.matvec(C.solve(x)) ≈ x for embed mode."""
    C = CirculantCovariance(embed_kernel, grid, mode="embed")
    x = rng.standard_normal(len(grid))
    np.testing.assert_allclose(C.matvec(C.solve(x)), x, atol=ATOL)


def test_solve_consistency_embed_2d(
    embed_kernel2d: StationaryKernel,
    grid2d: tuple[np.ndarray, np.ndarray],
    rng: Generator,
) -> None:
    C = CirculantCovariance(embed_kernel2d, grid2d, mode="embed")
    size = grid2d[0].size * grid2d[1].size
    x = rng.standard_normal(size)
    np.testing.assert_allclose(C.matvec(C.solve(x)), x, atol=ATOL)


def test_log_det_consistency_embed(
    embed_kernel: StationaryKernel,
    grid: NDArray[np.float64],
) -> None:
    """C.log_det() matches np.linalg.slogdet(C.to_dense()) for embed mode."""
    C = CirculantCovariance(embed_kernel, grid, mode="embed")
    _, ref = np.linalg.slogdet(C.to_dense())
    assert abs(C.log_det() - float(ref)) < ATOL * max(1.0, abs(float(ref)))


def test_log_det_consistency_embed_2d(
    embed_kernel2d: StationaryKernel,
    grid2d: tuple[np.ndarray, np.ndarray],
) -> None:
    C = CirculantCovariance(embed_kernel2d, grid2d, mode="embed")
    _, ref = np.linalg.slogdet(C.to_dense())
    assert abs(C.log_det() - float(ref)) < ATOL * max(1.0, abs(float(ref)))


def test_eigenvalues_nonneg(
    embed_kernel: StationaryKernel,
    grid: NDArray[np.float64],
) -> None:
    """All retained eigenvalues are non-negative."""
    C = CirculantCovariance(embed_kernel, grid, mode="embed")
    assert np.all(C.eigenvalues >= 0.0)


def test_eigenvalues_nonneg_2d(
    embed_kernel2d: StationaryKernel,
    grid2d: tuple[np.ndarray, np.ndarray],
) -> None:
    C = CirculantCovariance(embed_kernel2d, grid2d, mode="embed")
    assert np.all(C.eigenvalues >= 0.0)


def test_embed_dense_matches_toeplitz(
    embed_kernel: StationaryKernel,
    grid: NDArray[np.float64],
) -> None:
    """CirculantCovariance(embed).to_dense() ≈ ToeplitzCovariance.to_dense()."""
    C = CirculantCovariance(embed_kernel, grid, mode="embed")
    T = ToeplitzCovariance(embed_kernel, grid)
    np.testing.assert_allclose(C.to_dense(), T.to_dense(), atol=ATOL)


def test_embed_dense_matches_toeplitz_2d(
    embed_kernel2d: StationaryKernel,
    grid2d: tuple[np.ndarray, np.ndarray],
) -> None:
    C = CirculantCovariance(embed_kernel2d, grid2d, mode="embed")
    T = ToeplitzCovariance(embed_kernel2d, grid2d)
    np.testing.assert_allclose(C.to_dense(), T.to_dense(), atol=ATOL)


def test_invalid_embed_kernel_raises(
    invalid_embed_kernel: StationaryKernel,
    grid: NDArray[np.float64],
) -> None:
    with pytest.raises(ValueError, match="not positive semidefinite"):
        CirculantCovariance(invalid_embed_kernel, grid, mode="embed")


def test_invalid_embed_kernel_raises_2d(
    grid2d: tuple[np.ndarray, np.ndarray],
) -> None:
    def bad_kernel(r: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(r == 0.0, 1.0, 2.0)

    with pytest.raises(ValueError, match="not positive semidefinite"):
        CirculantCovariance(bad_kernel, grid2d, mode="embed")


# ---------------------------------------------------------------------------
# direct mode
# ---------------------------------------------------------------------------


def test_matvec_consistency_direct(
    embed_kernel: StationaryKernel,
    grid: NDArray[np.float64],
    rng: Generator,
) -> None:
    C = CirculantCovariance(embed_kernel, grid, mode="direct")
    x = rng.standard_normal(len(grid))
    np.testing.assert_allclose(C.matvec(x), C.to_dense() @ x, atol=ATOL)


def test_matvec_consistency_direct_2d(
    embed_kernel2d: StationaryKernel,
    grid2d: tuple[np.ndarray, np.ndarray],
    rng: Generator,
) -> None:
    C = CirculantCovariance(embed_kernel2d, grid2d, mode="direct")
    x = rng.standard_normal(grid2d[0].size * grid2d[1].size)
    np.testing.assert_allclose(C.matvec(x), C.to_dense() @ x, atol=ATOL)


def test_solve_consistency_direct(
    embed_kernel: StationaryKernel,
    grid: NDArray[np.float64],
    rng: Generator,
) -> None:
    C = CirculantCovariance(embed_kernel, grid, mode="direct")
    x = rng.standard_normal(len(grid))
    np.testing.assert_allclose(C.matvec(C.solve(x)), x, atol=ATOL)


def test_solve_consistency_direct_2d(
    embed_kernel2d: StationaryKernel,
    grid2d: tuple[np.ndarray, np.ndarray],
    rng: Generator,
) -> None:
    C = CirculantCovariance(embed_kernel2d, grid2d, mode="direct")
    size = grid2d[0].size * grid2d[1].size
    x = rng.standard_normal(size)
    np.testing.assert_allclose(C.matvec(C.solve(x)), x, atol=ATOL)


def test_log_det_consistency_direct(
    embed_kernel: StationaryKernel,
    grid: NDArray[np.float64],
) -> None:
    C = CirculantCovariance(embed_kernel, grid, mode="direct")
    _, ref = np.linalg.slogdet(C.to_dense())
    assert abs(C.log_det() - float(ref)) < ATOL * max(1.0, abs(float(ref)))


def test_log_det_consistency_direct_2d(
    embed_kernel2d: StationaryKernel,
    grid2d: tuple[np.ndarray, np.ndarray],
) -> None:
    C = CirculantCovariance(embed_kernel2d, grid2d, mode="direct")
    _, ref = np.linalg.slogdet(C.to_dense())
    assert abs(C.log_det() - float(ref)) < ATOL * max(1.0, abs(float(ref)))


def test_sample_covariance_matches_dense_embed(
    embed_kernel: StationaryKernel,
    grid: NDArray[np.float64],
    rng: Generator,
) -> None:
    C = CirculantCovariance(embed_kernel, grid, mode="embed")
    samples = C.sample(8_000, rng)
    empirical = np.cov(samples)
    np.testing.assert_allclose(empirical, C.to_dense(), atol=2.5e-1)


def test_sample_covariance_matches_dense_embed_2d(
    embed_kernel2d: StationaryKernel,
    grid2d: tuple[np.ndarray, np.ndarray],
    rng: Generator,
) -> None:
    C = CirculantCovariance(embed_kernel2d, grid2d, mode="embed")
    samples = C.sample(4_000, rng)
    empirical = np.cov(samples)
    np.testing.assert_allclose(empirical, C.to_dense(), atol=3.5e-1)


def test_sample_covariance_matches_dense_direct(
    embed_kernel: StationaryKernel,
    grid: NDArray[np.float64],
    rng: Generator,
) -> None:
    C = CirculantCovariance(embed_kernel, grid, mode="direct")
    samples = C.sample(8_000, rng)
    empirical = np.cov(samples)
    np.testing.assert_allclose(empirical, C.to_dense(), atol=2.5e-1)


def test_sample_covariance_matches_dense_direct_2d(
    embed_kernel2d: StationaryKernel,
    grid2d: tuple[np.ndarray, np.ndarray],
    rng: Generator,
) -> None:
    C = CirculantCovariance(embed_kernel2d, grid2d, mode="direct")
    samples = C.sample(4_000, rng)
    empirical = np.cov(samples)
    np.testing.assert_allclose(empirical, C.to_dense(), atol=3.5e-1)


def test_invalid_direct_kernel_raises(
    invalid_embed_kernel: StationaryKernel,
    grid: NDArray[np.float64],
) -> None:
    with pytest.raises(ValueError, match="not positive semidefinite"):
        CirculantCovariance(invalid_embed_kernel, grid, mode="direct")


def test_invalid_direct_kernel_raises_2d(
    grid2d: tuple[np.ndarray, np.ndarray],
) -> None:
    def bad_kernel(r: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(r == 0.0, 1.0, 2.0)

    with pytest.raises(ValueError, match="not positive semidefinite"):
        CirculantCovariance(bad_kernel, grid2d, mode="direct")


def test_nonuniform_grid_requires_opt_out(
    embed_kernel: StationaryKernel,
) -> None:
    grid = np.array([0.0, 0.2, 0.5, 0.9], dtype=np.float64)

    with pytest.raises(ValueError, match="uniformly spaced"):
        CirculantCovariance(embed_kernel, grid, mode="direct")

    C = CirculantCovariance(embed_kernel, grid, mode="direct", check_grid=False)
    assert C.to_dense().shape == (len(grid), len(grid))


def test_nonuniform_grid_requires_opt_out_2d(
    embed_kernel2d: StationaryKernel,
) -> None:
    grid = (
        np.array([0.0, 0.25, 0.7], dtype=np.float64),
        np.array([0.0, 0.4, 0.8], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="uniformly spaced"):
        CirculantCovariance(embed_kernel2d, grid, mode="direct")

    C = CirculantCovariance(embed_kernel2d, grid, mode="direct", check_grid=False)
    assert C.to_dense().shape == (9, 9)


def test_short_grid_raises(
    embed_kernel: StationaryKernel,
) -> None:
    with pytest.raises(ValueError, match="at least two points"):
        CirculantCovariance(
            embed_kernel, np.array([0.0], dtype=np.float64), mode="direct"
        )


def test_short_grid_raises_2d(
    embed_kernel2d: StationaryKernel,
) -> None:
    with pytest.raises(ValueError, match="at least two points"):
        CirculantCovariance(
            embed_kernel2d,
            (np.array([0.0], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64)),
            mode="direct",
        )


def test_invalid_embedding_raises() -> None:
    def bad_kernel(r: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(r == 0.0, 1.0, 2.0)

    grid = np.linspace(0.0, 1.0, 8)
    with pytest.raises(ValueError, match="not positive semidefinite"):
        CirculantCovariance(bad_kernel, grid, mode="embed")


def test_invalid_embedding_raises_2d() -> None:
    def bad_kernel(r: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(r == 0.0, 1.0, 2.0)

    grid = (np.linspace(0.0, 1.0, 6), np.linspace(0.0, 1.0, 5))
    with pytest.raises(ValueError, match="not positive semidefinite"):
        CirculantCovariance(bad_kernel, grid, mode="embed")


def test_embedding_tol_allows_clamping() -> None:
    def bad_kernel(r: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(r == 0.0, 1.0, 2.0)

    grid = np.linspace(0.0, 1.0, 8)
    C = CirculantCovariance(
        bad_kernel,
        grid,
        mode="embed",
        embedding_tol=16.0,
    )
    assert np.all(C.eigenvalues >= 0.0)


def test_embedding_tol_allows_clamping_2d() -> None:
    def bad_kernel(r: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(r == 0.0, 1.0, 2.0)

    grid = (np.linspace(0.0, 1.0, 6), np.linspace(0.0, 1.0, 5))
    C = CirculantCovariance(
        bad_kernel,
        grid,
        mode="embed",
        embedding_tol=64.0,
    )
    assert np.all(C.eigenvalues >= 0.0)
