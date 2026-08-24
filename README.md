# SteinVariational

[![Build Status](https://github.com/scheidan/SteinVariational.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/scheidan/SteinVariational.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/scheidan/SteinVariational.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/scheidan/SteinVariational.jl)


## Installation

From the Julia package manager:

```julia
] add https://github.com/scheidan/SteinVariational.jl
```

## Usage

The initial particle positions are given by a matrix where each row represents
a particle. A target can be supplied as separate functions for its log density
and gradient:

```julia
using SteinVariational

log_p(x) = -sum(abs2, x) / 2
∇log_p(x) = -x

inits = randn(100, 2) .+ 3
result = svgd(log_p, ∇log_p, inits; n_iter=500, stepsize=0.05)

result.particles
result.log_p
```

As in BarkerMCMC.jl, the first argument can instead be an object implementing
the first-order `LogDensityProblems.jl` interface:

```julia
result = svgd(lp, inits; n_iter=500, stepsize=0.05)
```

### Arguments

- `lp`: target implementing the first-order `LogDensityProblems.jl` interface.
- `log_p`: function returning the possibly unnormalized log density at a
  particle.
- `∇log_p`: function returning the gradient of `log_p` at a particle.
- `inits`: matrix whose rows contain the initial particle positions.
- `n_iter=100`: number of particle update iterations.
- `stepsize=0.1`: initial AdaGrad step size.
- `α=0.9`: decay applied to the AdaGrad history of squared update directions.
- `bandwidth=nothing`: RBF kernel bandwidth. The default applies the median
  heuristic at each iteration.
- `fudge_factor=1e-6`: value added to the AdaGrad denominator for numerical
  stability.
- `show_progress=true`: display a progress bar.

The return value is a named tuple with `particles`, containing the final
particles as rows, and `log_p`, containing their log densities.

## References

Liu, Q. & Wang, D. (2016). Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm. Advances in Neural Information Processing Systems 29.


Wang, D., Tang, Z., Bajaj, C. & Liu, Q. (2019). Stein Variational Gradient Descent with Matrix-Valued Kernels. NeurIPS 2019.
