from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def uniform_grid(a: float, b: float, n: int) -> NDArray[np.float64]:
    """Return n evenly-spaced points on [a, b]."""
    return np.linspace(a, b, n)


def validate_grid(
    grid: NDArray[np.float64],
    *,
    check_uniform: bool = True,
    uniform_tol: float = 1e-12,
) -> tuple[NDArray[np.float64], float]:
    """Validate a 1-D grid and return it as float64 plus a reference spacing."""
    grid64 = np.asarray(grid, dtype=np.float64)
    if grid64.ndim != 1:
        raise ValueError("grid must be one-dimensional")
    if grid64.size < 2:
        raise ValueError("grid must contain at least two points")

    diffs = np.diff(grid64)
    if not np.all(diffs > 0.0):
        raise ValueError("grid must be strictly increasing")

    h = float(diffs[0])
    if check_uniform and not np.allclose(diffs, h, atol=uniform_tol, rtol=uniform_tol):
        raise ValueError("grid must be uniformly spaced")

    return grid64, h


def check_positive_definite(
    lam: NDArray[np.float64], tol: float = 1e-10
) -> bool:
    """Return True iff all eigenvalues satisfy lam >= -tol."""
    return bool(np.all(lam >= -tol))
