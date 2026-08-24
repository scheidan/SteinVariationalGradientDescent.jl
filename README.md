# SteinVariationalGradientDescent

[![Build Status](https://github.com/scheidan/SteinVariationalGradientDescent.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/scheidan/SteinVariationalGradientDescent.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/scheidan/SteinVariationalGradientDescent.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/scheidan/SteinVariationalGradientDescent.jl)


Julia implementation of the Stein Variational Gradient Decent Algorithm as proposed
by Liu, Q. & Wang, D. (2016).

> [!WARNING]
> This package is a work in progress experimentation! Do not use it for anything important!

## Installation

From the Julia package manager:

```julia
] add https://github.com/scheidan/SteinVariationalGradientDescent.jl
```

## Usage

The initial particle positions are given by a matrix where each row represents
a particle. A target can be supplied as separate functions for its log density
and gradient:

```julia
using SteinVariationalGradientDescent

log_p(x) = -sum(abs2, x) / 2
∇log_p(x) = -x

inits = randn(100, 2) .+ 3
result = svgd(log_p, ∇log_p, inits; n_iter=1_000, stepsize=0.1)

result.particles
result.log_p
```

As in BarkerMCMC.jl, the first argument can instead be an object implementing
the first-order `LogDensityProblems.jl` interface:

```julia
result = svgd(lp, inits; n_iter=1_000, stepsize=0.1)
```

### Kernels

`RBFKernel()` is the default and gives the scalar-kernel SVGD algorithm of
Liu and Wang (2016). The current implementation is the special case
`K(x, x′) = k(x, x′)I` of matrix-valued SVGD.

`AverageMatrixRBFKernel(Q)` uses a single geometry matrix in the kernel. For
example, the precision matrix supplies the geometry for an anisotropic normal
target:

```julia
Q = [25.0 0.0; 0.0 0.25]
log_p(x) = -sum(x .* (Q * x)) / 2
∇log_p(x) = -(Q * x)

result = svgd(log_p, ∇log_p, inits;
              kernel=AverageMatrixRBFKernel(Q), n_iter=1_000)
```

The matrix can instead be calculated separately at every particle. The
average kernel averages these matrices, while the mixture kernel uses the
current particles as mixture anchors:

```julia
local_Q(x) = [4.0 0.0; 0.0 1.0 + abs(x[1])]

average_result = svgd(log_p, ∇log_p, inits;
                      kernel=AverageMatrixRBFKernel(local_Q))
mixture_result = svgd(log_p, ∇log_p, inits;
                      kernel=MixtureMatrixRBFKernel(local_Q))
```

Each supplied matrix must be symmetric positive definite! The package treats
the matrices as fixed while calculating one particle update and does not
differentiate through `Q(x)`.

### Arguments

- `lp`: target implementing the first-order `LogDensityProblems.jl` interface.
- `log_p`: function returning the possibly unnormalized log density at a
  particle.
- `∇log_p`: function returning the gradient of `log_p` at a particle.
- `inits`: matrix whose rows contain the initial particle positions.
- `n_iter=100`: number of particle update iterations.
- `stepsize=0.1`: particle update step size.
- `kernel=RBFKernel()`: kernel used to calculate the SVGD direction. The other
  choices are `AverageMatrixRBFKernel(Q)` and `MixtureMatrixRBFKernel(Q)`.
- `α=0.9`: decay applied to the AdaGrad history of squared `phi` values.
- `bandwidth=nothing`: RBF kernel bandwidth. The default applies the median
  heuristic at each iteration; the mixture kernel calculates a separate
  bandwidth for every component.
- `show_progress=true`: display a progress bar.

The kernel constructors take the following arguments:

- `RBFKernel()`: no arguments.
- `AverageMatrixRBFKernel(Q)`: `Q` is a fixed symmetric positive-definite
  matrix or a function returning one matrix for a particle. Function results
  are averaged at each iteration.
- `MixtureMatrixRBFKernel(Q)`: `Q` has the same forms, but supplies one geometry
  matrix per current-particle anchor.

The return value is a named tuple with `particles`, containing the final
particles as rows, and `log_p`, containing their log densities.

## References

Liu, Q. & Wang, D. (2016). Stein Variational Gradient Descent: A General
Purpose Bayesian Inference Algorithm. Advances in Neural Information
Processing Systems 29.

Wang, D., Tang, Z., Bajaj, C. & Liu, Q. (2019). Stein Variational Gradient
Descent with Matrix-Valued Kernels. Advances in Neural Information Processing
Systems 32.
