# circcov
A basic implementation for covariance matrices, accelerated using the circulant approximation

## API reference

- `CirculantCovariance(kernel, grid, mode="embed", *, jitter=1e-6, embedding_tol=1e-8, check_grid=True)`: circulant approximation on a 1-D uniform grid. Main methods: `matvec(x)`, `solve(x)`, `log_det()`, `diagonal()`, `sample(n=1, rng=None)`, `to_dense()`, and `eigenvalues`.
- `ToeplitzCovariance(kernel, grid, *, jitter=1e-6, check_grid=True)`: exact dense Toeplitz covariance on a 1-D uniform grid. Main methods: `matvec(x)`, `solve(x)`, `log_det()`, `diagonal()`, `sample(n=1, rng=None)`, and `to_dense()`.
- `matern(nu, length_scale=1.0, variance=1.0)`: returns a stationary Matérn kernel. Supported `nu` values are `0.5`, `1.5`, `2.5`, or `>= 10` for the squared-exponential limit.
- `squared_exponential(length_scale=1.0, variance=1.0)`: returns a stationary RBF kernel.
- `uniform_grid(a, b, n)`: returns `n` evenly spaced points on `[a, b]`.
