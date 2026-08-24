# SteinVariational

[![Build Status](https://github.com/scheidan/SteinVariational.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/scheidan/SteinVariational.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/scheidan/SteinVariational.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/scheidan/SteinVariational.jl)


## Installation

From the Julia package manager:

```julia
] add https://github.com/scheidan/SteinVariational.jl
```

## Usage

The initial particle positions are given by a matrix where each row is
represents a particle.
A target can be supplied as a
pair of functions for its log density and gradient:

```julia
using SteinVariational

log_p(x) = -sum(abs2, x) / 2
∇log_p(x) = -x

inits = randn(100, 2) .+ 3
result = svgd(log_p, ∇log_p, inits; n_iter=500, stepsize=0.05)

result.samples
result.log_p
```

As in BarkerMCMC.jl, the first argument can instead be an object implementing
the first-order `LogDensityProblems.jl` interface:

```julia
result = svgd(lp, inits; n_iter=500, stepsize=0.05)
```

## References

Liu, Q. & Wang, D. (2016). Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm. Advances in Neural Information Processing Systems 29.


Wang, D., Tang, Z., Bajaj, C. & Liu, Q. (2019). Stein Variational Gradient Descent with Matrix-Valued Kernels. NeurIPS 2019.
