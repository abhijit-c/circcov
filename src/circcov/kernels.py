from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

LengthScale: TypeAlias = float | tuple[float, float]


class StationaryKernel(Protocol):
    def __call__(self, r: NDArray[np.float64]) -> NDArray[np.float64]: ...


def _normalize_length_scale(length_scale: LengthScale) -> float | tuple[float, float]:
    arr = np.asarray(length_scale, dtype=np.float64)
    if arr.ndim == 0:
        value = float(arr)
        if value <= 0.0:
            raise ValueError("length_scale must be positive")
        return value

    if arr.ndim != 1 or arr.size != 2:
        raise ValueError("length_scale must be a positive scalar or length-2 tuple")

    values = tuple(float(value) for value in arr)
    if values[0] <= 0.0 or values[1] <= 0.0:
        raise ValueError("length_scale must be positive")
    return values


@dataclass(frozen=True)
class _ScaledStationaryKernel:
    length_scale: float | tuple[float, float]
    variance: float

    def _direct_scale(self) -> float:
        if isinstance(self.length_scale, tuple):
            if not math.isclose(self.length_scale[0], self.length_scale[1]):
                raise ValueError(
                    "anisotropic kernels require scaled radial distances"
                )
            return self.length_scale[0]
        return self.length_scale

    def evaluate_scaled(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError

    def __call__(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        r64 = np.asarray(r, dtype=np.float64)
        scale = self._direct_scale()
        return self.evaluate_scaled(r64 / scale)


@dataclass(frozen=True)
class _MaternKernel(_ScaledStationaryKernel):
    nu: float

    def evaluate_scaled(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.nu == 0.5:
            return self.variance * np.exp(-r)  # type: ignore[return-value]

        if self.nu == 1.5:
            u = math.sqrt(3.0) * r
            return self.variance * (1.0 + u) * np.exp(-u)  # type: ignore[return-value]

        u = math.sqrt(5.0) * r
        return self.variance * (1.0 + u + u**2 / 3.0) * np.exp(-u)  # type: ignore[return-value]


@dataclass(frozen=True)
class _SquaredExponentialKernel(_ScaledStationaryKernel):
    def evaluate_scaled(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.variance * np.exp(-(r**2) / 2.0)  # type: ignore[return-value]


def matern(
    nu: float, length_scale: LengthScale = 1.0, variance: float = 1.0
) -> StationaryKernel:
    """Return a Matérn kernel with smoothness nu.

    Supported nu: 0.5, 1.5, 2.5. nu >= 10 is treated as the
    squared-exponential (nu -> infinity) limit.
    """
    normalized_scale = _normalize_length_scale(length_scale)

    if nu >= 10:
        return squared_exponential(length_scale=normalized_scale, variance=variance)

    supported = {0.5, 1.5, 2.5}
    if nu not in supported:
        raise ValueError(f"nu must be one of {supported} or >= 10, got {nu}")

    return _MaternKernel(length_scale=normalized_scale, variance=variance, nu=nu)


def squared_exponential(
    length_scale: LengthScale = 1.0, variance: float = 1.0
) -> StationaryKernel:
    """Return a squared-exponential (RBF / Gaussian) kernel."""
    normalized_scale = _normalize_length_scale(length_scale)
    return _SquaredExponentialKernel(
        length_scale=normalized_scale,
        variance=variance,
    )
