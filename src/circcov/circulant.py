from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, cg  # type: ignore[import-untyped]

from .kernels import StationaryKernel
from .utils import validate_grid

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


class CirculantCovariance:
    """Circulant approximation of a stationary covariance matrix.

    Parameters
    ----------
    kernel:
        A stationary kernel k(r).
    grid:
        Uniform 1-D grid of n points.
    mode:
        ``"embed"`` (default) — Wood-Chan embedding: embed the n×n Toeplitz
        matrix in a symmetric circulant of size ``2 * (n - 1)`` and use FFT
        matvecs on its leading principal block.
        ``"direct"`` — form an n×n symmetric circulant by wrapping the kernel
        row on the periodic domain.  Fast approximation; not generally exact.
    """

    def __init__(
        self,
        kernel: StationaryKernel,
        grid: NDArray[np.float64],
        mode: str = "embed",
        *,
        jitter: float = _REGULARIZATION,
        embedding_tol: float = _NEG_TOL,
        check_grid: bool = True,
    ) -> None:
        if mode not in ("embed", "direct"):
            raise ValueError(f"mode must be 'embed' or 'direct', got {mode!r}")

        grid64, h = validate_grid(grid, check_uniform=check_grid)
        n = grid64.size
        self._grid = grid64
        self._n = n
        self._mode = mode
        self._h = h
        self._jitter = float(jitter)
        self._embedding_tol = float(embedding_tol)
        self._dense_cache: NDArray[np.float64] | None = None

        if mode == "embed":
            row = np.asarray(
                kernel(np.arange(n, dtype=np.float64) * h), dtype=np.float64
            )
            self._row = row
            self._m = 2 * (n - 1)
            self._circulant_row = np.concatenate((row, row[-2:0:-1]))
        else:
            wrap_lags = np.minimum(
                np.arange(n, dtype=np.float64),
                np.arange(n, 0, -1, dtype=np.float64),
            )
            self._row = np.asarray(kernel(wrap_lags * h), dtype=np.float64)
            self._m = n
            self._circulant_row = self._row.copy()

        self._circulant_row[0] += self._jitter
        lam = np.fft.rfft(self._circulant_row).real
        worst = float(lam.min())
        if worst < -self._embedding_tol:
            label = "Wood-Chan embedding" if mode == "embed" else "circulant spectrum"
            raise ValueError(
                f"{label} is not positive semidefinite: min eigenvalue = {worst:.3e}"
            )
        self._lam: NDArray[np.float64] = np.maximum(lam, 0.0)
        self._solve_rtol = 1e-10
        self._solve_maxiter = max(10 * self._n, 100)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def eigenvalues(self) -> NDArray[np.float64]:
        """Eigenvalues of the circulant (non-negative after clamping in embed mode)."""
        return self._lam

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Multiply the represented covariance matrix by vector x."""
        x64 = np.asarray(x, dtype=np.float64)
        if x64.shape != (self._n,):
            raise ValueError(f"x must have shape ({self._n},)")

        if self._mode == "direct":
            y = np.fft.irfft(self._lam * np.fft.rfft(x64), n=self._m)
            return np.asarray(y, dtype=np.float64)

        x_pad = np.zeros(self._m, dtype=np.float64)
        x_pad[: self._n] = x64
        y_pad = np.fft.irfft(self._lam * np.fft.rfft(x_pad), n=self._m)
        return np.asarray(y_pad[: self._n], dtype=np.float64)

    def solve(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve C @ result = x."""
        x64 = np.asarray(x, dtype=np.float64)
        if x64.shape != (self._n,):
            raise ValueError(f"x must have shape ({self._n},)")

        if self._mode == "direct":
            y = np.fft.irfft(np.fft.rfft(x64) / self._lam, n=self._m)
            return np.asarray(y, dtype=np.float64)

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
            weights = np.ones_like(self._lam)
            if self._m % 2 == 0:
                weights[1:-1] = 2.0
            else:
                weights[1:] = 2.0
            return float(np.sum(weights * np.log(self._lam)))

        sign, value = np.linalg.slogdet(self.to_dense())
        if sign <= 0:
            raise ValueError("represented covariance matrix is not positive definite")
        return float(value)

    def diagonal(self) -> NDArray[np.float64]:
        """Diagonal of the represented covariance matrix."""
        return np.full(self._n, self._row[0] + self._jitter, dtype=np.float64)

    def sample(
        self, n: int = 1, rng: Generator | None = None
    ) -> NDArray[np.float64]:
        """Draw n independent samples from N(0, C).

        Returns an array of shape (self._n, n).
        """
        if rng is None:
            rng = np.random.default_rng()

        noise = rng.standard_normal((self._m, n))
        spectral_noise = np.fft.rfft(noise, axis=0)
        scaled = np.sqrt(self._lam)[:, None] * spectral_noise
        samples = np.fft.irfft(scaled, n=self._m, axis=0)
        return np.asarray(samples[: self._n, :], dtype=np.float64)

    def to_dense(self) -> NDArray[np.float64]:
        """Return the represented covariance matrix as a dense array."""
        if self._dense_cache is None:
            idx = np.arange(self._n)
            if self._mode == "direct":
                dense = self._circulant_row[(idx[None, :] - idx[:, None]) % self._n]
            else:
                dense = self._row[np.abs(idx[:, None] - idx[None, :])].copy()
                dense[np.diag_indices_from(dense)] += self._jitter
            self._dense_cache = np.asarray(dense, dtype=np.float64)
        return self._dense_cache.copy()
