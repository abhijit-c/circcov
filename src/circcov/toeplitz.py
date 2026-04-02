from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve  # type: ignore[import-untyped]

from .kernels import StationaryKernel
from .utils import validate_grid

_REGULARIZATION = 1e-6


class ToeplitzCovariance:
    """Exact Toeplitz covariance matrix for a stationary kernel on a uniform grid.

    Intended as a reference implementation for correctness testing.  Uses a
    dense Cholesky factorisation, so is O(n³) to construct and O(n²) per solve.
    """

    def __init__(
        self,
        kernel: StationaryKernel,
        grid: NDArray[np.float64],
        *,
        jitter: float = _REGULARIZATION,
        check_grid: bool = True,
    ) -> None:
        grid64, _ = validate_grid(grid, check_uniform=check_grid)
        self._grid = grid64
        self._n = grid64.size
        self._jitter = float(jitter)

        distances = np.abs(grid64[:, None] - grid64[None, :])
        self._mat: NDArray[np.float64] = np.asarray(
            kernel(distances), dtype=np.float64
        )
        self._mat[np.diag_indices_from(self._mat)] += self._jitter
        self._cho = cho_factor(self._mat, lower=False)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Multiply the covariance matrix by vector x."""
        return self._mat @ x  # type: ignore[return-value]

    def solve(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve C @ result = x via Cholesky."""
        return cho_solve(self._cho, x)  # type: ignore[return-value]

    def log_det(self) -> float:
        """Log-determinant via Cholesky: 2 * sum(log(diag(L)))."""
        L = self._cho[0]
        return float(2.0 * np.sum(np.log(np.diag(L))))

    def diagonal(self) -> NDArray[np.float64]:
        """Diagonal entries (all equal k(0) for a stationary kernel)."""
        return np.diag(self._mat).copy()

    def sample(
        self, n: int = 1, rng: Generator | None = None
    ) -> NDArray[np.float64]:
        """Draw n independent samples from N(0, C).

        Returns an array of shape (self._n, n).
        """
        if rng is None:
            rng = np.random.default_rng()

        size = self._n
        upper = np.triu(self._cho[0])
        z = rng.standard_normal((size, n))
        return upper.T @ z

    def to_dense(self) -> NDArray[np.float64]:
        """Return the covariance matrix as a dense array."""
        return self._mat.copy()
