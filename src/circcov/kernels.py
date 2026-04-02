from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class StationaryKernel(Protocol):
    def __call__(self, r: NDArray[np.float64]) -> NDArray[np.float64]: ...


def matern(
    nu: float, length_scale: float = 1.0, variance: float = 1.0
) -> StationaryKernel:
    """Return a Matérn kernel with smoothness nu.

    Supported nu: 0.5, 1.5, 2.5.  nu >= 10 is treated as the
    squared-exponential (nu -> infinity) limit.
    """
    if nu >= 10:
        return squared_exponential(length_scale=length_scale, variance=variance)

    supported = {0.5, 1.5, 2.5}
    if nu not in supported:
        raise ValueError(f"nu must be one of {supported} or >= 10, got {nu}")

    if nu == 0.5:
        def _k05(r: NDArray[np.float64]) -> NDArray[np.float64]:
            return variance * np.exp(-r / length_scale)  # type: ignore[return-value]
        return _k05

    if nu == 1.5:
        sqrt3 = math.sqrt(3.0)

        def _k15(r: NDArray[np.float64]) -> NDArray[np.float64]:
            u = sqrt3 * r / length_scale
            return variance * (1.0 + u) * np.exp(-u)  # type: ignore[return-value]
        return _k15

    # nu == 2.5
    sqrt5 = math.sqrt(5.0)

    def _k25(r: NDArray[np.float64]) -> NDArray[np.float64]:
        u = sqrt5 * r / length_scale
        return variance * (1.0 + u + u**2 / 3.0) * np.exp(-u)  # type: ignore[return-value]
    return _k25


def squared_exponential(
    length_scale: float = 1.0, variance: float = 1.0
) -> StationaryKernel:
    """Return a squared-exponential (RBF / Gaussian) kernel."""

    def _kse(r: NDArray[np.float64]) -> NDArray[np.float64]:
        return variance * np.exp(-(r**2) / (2.0 * length_scale**2))  # type: ignore[return-value]

    return _kse
