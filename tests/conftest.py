from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator

from circcov import matern, squared_exponential, uniform_grid
from circcov.kernels import StationaryKernel


@pytest.fixture(params=[32, 64])
def n(request: pytest.FixtureRequest) -> int:
    return int(request.param)


@pytest.fixture
def grid(n: int) -> np.ndarray:
    return uniform_grid(0.0, 1.0, n)


@pytest.fixture(params=[(6, 5), (7, 4)], ids=["grid2d_6x5", "grid2d_7x4"])
def grid2d(request: pytest.FixtureRequest) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = request.param
    return (
        uniform_grid(0.0, 1.0, nx),
        uniform_grid(-0.5, 0.5, ny),
    )


@pytest.fixture(
    params=["matern_05", "matern_15", "matern_25", "se"],
    ids=["matern_05", "matern_15", "matern_25", "se"],
)
def kernel(request: pytest.FixtureRequest) -> StationaryKernel:
    name: str = request.param
    if name == "matern_05":
        return matern(0.5, length_scale=0.3, variance=2.0)
    if name == "matern_15":
        return matern(1.5, length_scale=0.3, variance=2.0)
    if name == "matern_25":
        return matern(2.5, length_scale=0.3, variance=2.0)
    return squared_exponential(length_scale=0.3, variance=2.0)


@pytest.fixture
def kernel2d() -> StationaryKernel:
    return matern(1.5, length_scale=(0.3, 0.45), variance=2.0)


@pytest.fixture(params=["matern_05"], ids=["matern_05"])
def embed_kernel(request: pytest.FixtureRequest) -> StationaryKernel:
    name: str = request.param
    if name == "matern_05":
        return matern(0.5, length_scale=0.3, variance=2.0)
    raise AssertionError(f"unexpected embed kernel {name}")


@pytest.fixture
def embed_kernel2d() -> StationaryKernel:
    return matern(0.5, length_scale=(0.25, 0.4), variance=2.0)


@pytest.fixture(
    params=["matern_15", "matern_25", "se"],
    ids=["matern_15", "matern_25", "se"],
)
def invalid_embed_kernel(request: pytest.FixtureRequest) -> StationaryKernel:
    name: str = request.param
    if name == "matern_15":
        return matern(1.5, length_scale=0.3, variance=2.0)
    if name == "matern_25":
        return matern(2.5, length_scale=0.3, variance=2.0)
    return squared_exponential(length_scale=0.3, variance=2.0)


@pytest.fixture
def invalid_embed_kernel2d() -> StationaryKernel:
    return matern(1.5, length_scale=(0.3, 0.45), variance=2.0)


@pytest.fixture
def rng() -> Generator:
    return np.random.default_rng(42)
