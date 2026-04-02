from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve  # type: ignore[import-untyped]
from scipy.sparse.linalg import LinearOperator, cg  # type: ignore[import-untyped]

from .kernels import StationaryKernel
from .utils import (
    GridLike,
    evaluate_kernel_offsets,
    flatten_grid_values,
    unflatten_grid_values,
    validate_grid,
)

_NEG_TOL = 1e-8
_REGULARIZATION = 1e-6


class _FFTLinearOperator(LinearOperator):
    def __init__(
        self,
        n: int,
        matvec_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> None:
        super().__init__(dtype=np.float64, shape=(n, n))
        self._matvec_fn = matvec_fn

    def _matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._matvec_fn(np.asarray(x, dtype=np.float64))


def _mirror_embed_axis(values: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    if values.shape[axis] == 2:
        mirrored = np.take(values, [0], axis=axis)
    else:
        mirrored = np.flip(
            np.take(values, np.arange(1, values.shape[axis] - 1), axis=axis),
            axis=axis,
        )
    return np.concatenate((values, mirrored), axis=axis)


class CirculantCovariance:
    """Circulant approximation of a stationary covariance matrix."""

    def __init__(
        self,
        kernel: StationaryKernel,
        grid: GridLike,
        mode: str = "embed",
        *,
        jitter: float = _REGULARIZATION,
        embedding_tol: float = _NEG_TOL,
        check_grid: bool = True,
    ) -> None:
        if mode not in ("embed", "direct"):
            raise ValueError(f"mode must be 'embed' or 'direct', got {mode!r}")

        spec = validate_grid(grid, check_uniform=check_grid)
        self._grid = grid
        self._shape = spec.shape
        self._ndim = spec.ndim
        self._n = spec.size
        self._mode = mode
        self._jitter = float(jitter)
        self._embedding_tol = float(embedding_tol)
        self._dense_cache: NDArray[np.float64] | None = None
        self._sample_cho: tuple[NDArray[np.float64], bool] | None = None

        lag_axes: list[NDArray[np.float64]] = []
        for axis, spacing, size in zip(spec.axes, spec.spacings, spec.shape, strict=True):
            if mode == "embed":
                lag_axes.append(axis - axis[0])
            else:
                indices = np.minimum(
                    np.arange(size, dtype=np.float64),
                    size - np.arange(size, dtype=np.float64),
                )
                lag_axes.append(indices * spacing)

        if spec.ndim == 1:
            offsets = lag_axes[0][:, None]
        else:
            x_lags, y_lags = np.meshgrid(*lag_axes, indexing="ij")
            offsets = np.stack((x_lags, y_lags), axis=-1)

        self._row = np.asarray(evaluate_kernel_offsets(kernel, offsets), dtype=np.float64)
        self._mshape = self._row.shape
        self._circulant_row = self._build_circulant_row(self._row, mode)
        self._circulant_row[(0,) * self._circulant_row.ndim] += self._jitter

        lam = np.fft.fftn(self._circulant_row).real
        self._lam_rfft: NDArray[np.float64] | None = None
        if self._ndim == 1:
            self._lam_rfft = np.fft.rfft(self._circulant_row).real
        worst = float(lam.min())
        if worst < -self._embedding_tol:
            label = "Wood-Chan embedding" if mode == "embed" else "circulant spectrum"
            raise ValueError(
                f"{label} is not positive semidefinite: min eigenvalue = {worst:.3e}"
            )
        self._lam: NDArray[np.float64] = np.maximum(lam, 0.0)
        self._solve_rtol = 1e-10
        self._solve_maxiter = max(10 * self._n, 100)

    @staticmethod
    def _build_circulant_row(row: NDArray[np.float64], mode: str) -> NDArray[np.float64]:
        if mode == "direct":
            return row.copy()

        circ = row.copy()
        for axis in range(circ.ndim):
            circ = _mirror_embed_axis(circ, axis)
        return circ

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def eigenvalues(self) -> NDArray[np.float64]:
        """Eigenvalues of the circulant after clamping."""
        return self._lam.copy()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def _apply_fft(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        spectrum = np.fft.fftn(values)
        result = np.fft.ifftn(self._lam * spectrum)
        return np.asarray(result.real, dtype=np.float64)

    def matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Multiply the represented covariance matrix by vector x."""
        x64 = np.asarray(x, dtype=np.float64)
        if x64.shape != (self._n,):
            raise ValueError(f"x must have shape ({self._n},)")

        if self._ndim == 1 and self._lam_rfft is not None:
            if self._mode == "direct":
                y = np.fft.irfft(self._lam_rfft * np.fft.rfft(x64), n=self._shape[0])
                return np.asarray(y, dtype=np.float64)

            padded = np.zeros(self._circulant_row.shape[0], dtype=np.float64)
            padded[: self._n] = x64
            y = np.fft.irfft(
                self._lam_rfft * np.fft.rfft(padded),
                n=self._circulant_row.shape[0],
            )
            return np.asarray(y[: self._n], dtype=np.float64)

        if self._mode == "direct":
            shaped = unflatten_grid_values(x64, self._shape)
            y = self._apply_fft(shaped)
            return flatten_grid_values(y, self._shape)

        padded = np.zeros(self._circulant_row.shape, dtype=np.float64)
        padded[tuple(slice(0, size) for size in self._shape)] = unflatten_grid_values(
            x64, self._shape
        )
        y = self._apply_fft(padded)
        leading = y[tuple(slice(0, size) for size in self._shape)]
        return flatten_grid_values(leading, self._shape)

    def solve(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve C @ result = x."""
        x64 = np.asarray(x, dtype=np.float64)
        if x64.shape != (self._n,):
            raise ValueError(f"x must have shape ({self._n},)")

        if self._ndim == 1 and self._lam_rfft is not None and self._mode == "direct":
            y = np.fft.irfft(np.fft.rfft(x64) / self._lam_rfft, n=self._shape[0])
            return np.asarray(y, dtype=np.float64)

        if self._mode == "direct":
            shaped = unflatten_grid_values(x64, self._shape)
            y = np.fft.ifftn(np.fft.fftn(shaped) / self._lam).real
            return flatten_grid_values(np.asarray(y, dtype=np.float64), self._shape)

        operator = _FFTLinearOperator(self._n, self.matvec)
        result, info = cg(
            operator,
            x64,
            rtol=self._solve_rtol,
            atol=0.0,
            maxiter=self._solve_maxiter,
        )
        if info != 0:
            raise RuntimeError(f"CG solve did not converge (info={info})")
        return np.asarray(result, dtype=np.float64)

    def log_det(self) -> float:
        """Log-determinant of the represented covariance matrix."""
        if self._mode == "direct":
            if self._ndim == 1 and self._lam_rfft is not None:
                weights = np.ones_like(self._lam_rfft)
                if self._shape[0] % 2 == 0:
                    weights[1:-1] = 2.0
                else:
                    weights[1:] = 2.0
                return float(np.sum(weights * np.log(self._lam_rfft)))
            return float(np.sum(np.log(self._lam)))

        sign, value = np.linalg.slogdet(self.to_dense())
        if sign <= 0:
            raise ValueError("represented covariance matrix is not positive definite")
        return float(value)

    def diagonal(self) -> NDArray[np.float64]:
        """Diagonal of the represented covariance matrix."""
        return np.full(self._n, self._row[(0,) * self._row.ndim] + self._jitter, dtype=np.float64)

    def sample(
        self, n: int = 1, rng: Generator | None = None
    ) -> NDArray[np.float64]:
        """Draw n independent samples from N(0, C)."""
        if rng is None:
            rng = np.random.default_rng()

        if self._sample_cho is None:
            self._sample_cho = cho_factor(self.to_dense(), lower=False)

        z = rng.standard_normal((self._n, n))
        upper = np.triu(self._sample_cho[0])
        return upper.T @ z

    def to_dense(self) -> NDArray[np.float64]:
        """Return the represented covariance matrix as a dense array."""
        if self._dense_cache is None:
            coords = np.column_stack(
                np.unravel_index(np.arange(self._n), self._shape, order="F")
            )
            if self._mode == "direct":
                indexers = tuple(
                    (coords[None, :, axis] - coords[:, None, axis]) % self._shape[axis]
                    for axis in range(self._ndim)
                )
                dense = self._row[indexers].copy()
                dense[np.diag_indices_from(dense)] += self._jitter
            else:
                indexers = tuple(
                    np.abs(coords[:, None, axis] - coords[None, :, axis])
                    for axis in range(self._ndim)
                )
                dense = self._row[indexers].copy()
                dense[np.diag_indices_from(dense)] += self._jitter
            self._dense_cache = np.asarray(dense, dtype=np.float64)
        return self._dense_cache.copy()
