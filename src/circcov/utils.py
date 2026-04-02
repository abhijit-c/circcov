from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .kernels import StationaryKernel

GridLike = NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]


@dataclass(frozen=True)
class GridSpec:
    axes: tuple[NDArray[np.float64], ...]
    spacings: tuple[float, ...]
    shape: tuple[int, ...]
    points: NDArray[np.float64]

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))


def uniform_grid(a: float, b: float, n: int) -> NDArray[np.float64]:
    """Return n evenly-spaced points on [a, b]."""
    return np.linspace(a, b, n)


def _validate_axis(
    axis: NDArray[np.float64],
    *,
    check_uniform: bool = True,
    uniform_tol: float = 1e-12,
) -> tuple[NDArray[np.float64], float]:
    axis64 = np.asarray(axis, dtype=np.float64)
    if axis64.ndim != 1:
        raise ValueError("grid axes must be one-dimensional")
    if axis64.size < 2:
        raise ValueError("grid must contain at least two points per axis")

    diffs = np.diff(axis64)
    if not np.all(diffs > 0.0):
        raise ValueError("grid axes must be strictly increasing")

    h = float(diffs[0])
    if check_uniform and not np.allclose(diffs, h, atol=uniform_tol, rtol=uniform_tol):
        raise ValueError("grid must be uniformly spaced")

    return axis64, h


def validate_grid(
    grid: GridLike,
    *,
    check_uniform: bool = True,
    uniform_tol: float = 1e-12,
) -> GridSpec:
    """Validate a 1-D grid or 2-D tensor grid."""
    if isinstance(grid, tuple):
        if len(grid) != 2:
            raise ValueError("2-D grids must be provided as (x_grid, y_grid)")
        axes_and_steps = tuple(
            _validate_axis(
                axis,
                check_uniform=check_uniform,
                uniform_tol=uniform_tol,
            )
            for axis in grid
        )
    else:
        axes_and_steps = (
            _validate_axis(
                grid,
                check_uniform=check_uniform,
                uniform_tol=uniform_tol,
            ),
        )

    axes = tuple(axis for axis, _ in axes_and_steps)
    spacings = tuple(step for _, step in axes_and_steps)
    shape = tuple(axis.size for axis in axes)

    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack(
        [np.ravel(component, order="F") for component in mesh]
    ).astype(np.float64, copy=False)
    return GridSpec(axes=axes, spacings=spacings, shape=shape, points=points)


def flatten_grid_values(
    values: NDArray[np.float64],
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    values64 = np.asarray(values, dtype=np.float64)
    if values64.shape != shape:
        raise ValueError(f"values must have shape {shape}")
    return np.reshape(values64, -1, order="F")


def unflatten_grid_values(
    values: NDArray[np.float64],
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    values64 = np.asarray(values, dtype=np.float64)
    if values64.shape != (int(np.prod(shape)),):
        raise ValueError(f"values must have shape ({int(np.prod(shape))},)")
    return np.reshape(values64, shape, order="F")


def evaluate_kernel_offsets(
    kernel: StationaryKernel,
    offsets: NDArray[np.float64],
) -> NDArray[np.float64]:
    offsets64 = np.asarray(offsets, dtype=np.float64)
    if offsets64.shape[-1] == 0:
        raise ValueError("offsets must include at least one coordinate axis")

    raw_radii = np.linalg.norm(offsets64, axis=-1)
    evaluate_scaled = getattr(kernel, "evaluate_scaled", None)
    length_scale = getattr(kernel, "length_scale", None)

    if callable(evaluate_scaled) and isinstance(length_scale, tuple):
        scale = np.asarray(length_scale, dtype=np.float64)
        scaled = np.linalg.norm(offsets64 / scale, axis=-1)
        return np.asarray(evaluate_scaled(scaled), dtype=np.float64)

    return np.asarray(kernel(raw_radii), dtype=np.float64)


def check_positive_definite(
    lam: NDArray[np.float64], tol: float = 1e-10
) -> bool:
    """Return True iff all eigenvalues satisfy lam >= -tol."""
    return bool(np.all(lam >= -tol))
