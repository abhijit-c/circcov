# circcov
A basic implementation for covariance matrices, accelerated using the circulant approximation

## API reference

- `CirculantCovariance(kernel, grid, mode="embed", *, jitter=1e-6, embedding_tol=1e-8, check_grid=True)`: circulant approximation on a 1-D uniform grid or 2-D tensor grid `grid=(x_grid, y_grid)`. Main methods: `matvec(x)`, `solve(x)`, `log_det()`, `diagonal()`, `sample(n=1, rng=None)`, `to_dense()`, and `eigenvalues`.
- `ToeplitzCovariance(kernel, grid, *, jitter=1e-6, check_grid=True)`: exact dense Toeplitz covariance on a 1-D uniform grid or dense block-Toeplitz-with-Toeplitz-blocks covariance on a 2-D tensor grid.
- `matern(nu, length_scale=1.0, variance=1.0)`: returns a stationary Matérn kernel. `length_scale` may be a scalar or a 2-tuple for anisotropic 2-D tensor grids. Supported `nu` values are `0.5`, `1.5`, `2.5`, or `>= 10` for the squared-exponential limit.
- `squared_exponential(length_scale=1.0, variance=1.0)`: returns a stationary RBF kernel. `length_scale` may be a scalar or a 2-tuple.
- `uniform_grid(a, b, n)`: returns `n` evenly spaced points on `[a, b]`.

For 2-D grids, vector-valued operations use flattened arrays of length `len(x_grid) * len(y_grid)` in Fortran order.
