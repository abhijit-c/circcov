from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve  # type: ignore[import-untyped]

from .kernels import StationaryKernel
from .utils import GridLike, evaluate_kernel_offsets, validate_grid

_REGULARIZATION = 1e-6


class ToeplitzCovariance:
    """Exact covariance matrix for a stationary kernel on a uniform grid.

    For 1-D inputs this is Toeplitz. For 2-D tensor grids this is the dense
    block-Toeplitz-with-Toeplitz-blocks reference operator.
    """

    def __init__(
        self,
        kernel: StationaryKernel,
        grid: GridLike,
        *,
        jitter: float = _REGULARIZATION,
        check_grid: bool = True,
    ) -> None:
        spec = validate_grid(grid, check_uniform=check_grid)
        self._grid = grid
        self._shape = spec.shape
        self._n = spec.size
        self._jitter = float(jitter)

        offsets = spec.points[:, None, :] - spec.points[None, :, :]
        self._mat: NDArray[np.float64] = evaluate_kernel_offsets(kernel, offsets)
        self._mat[np.diag_indices_from(self._mat)] += self._jitter
        self._cho = cho_factor(self._mat, lower=False)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Multiply the covariance matrix by vector x."""
        x64 = np.asarray(x, dtype=np.float64)
        if x64.shape != (self._n,):
            raise ValueError(f"x must have shape ({self._n},)")
        return self._mat @ x64  # type: ignore[return-value]

    def solve(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve C @ result = x via Cholesky."""
        x64 = np.asarray(x, dtype=np.float64)
        if x64.shape != (self._n,):
            raise ValueError(f"x must have shape ({self._n},)")
        return cho_solve(self._cho, x64)  # type: ignore[return-value]

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

        upper = np.triu(self._cho[0])
        z = rng.standard_normal((self._n, n))
        return upper.T @ z

    def to_dense(self) -> NDArray[np.float64]:
        """Return the covariance matrix as a dense array."""
        return self._mat.copy()
