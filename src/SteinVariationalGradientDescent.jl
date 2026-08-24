module SteinVariationalGradientDescent

export AverageMatrixRBFKernel, MixtureMatrixRBFKernel, RBFKernel, svgd

using LinearAlgebra: cholesky, logdet
using Statistics: median
import LogDensityProblems
import ProgressMeter

include("logdensityproblem_interface.jl")
include("kernels.jl")

"""
Approximate a target distribution with Stein variational gradient descent.

```julia
    svgd(lp, inits;
         n_iter=100, stepsize=0.1, kernel=RBFKernel(), bandwidth=nothing,
         α=0.9, show_progress=true)
```
or:
```julia
    svgd(log_p, ∇log_p, inits;
         n_iter=100, stepsize=0.1, kernel=RBFKernel(), bandwidth=nothing,
         α=0.9, show_progress=true)
```

The rows of `inits` are the initial particles. The target can be a
`LogDensityProblems.jl` object that provides first derivatives, or it can be
given as separate log-density and gradient functions. `RBFKernel()` gives
standard SVGD. `AverageMatrixRBFKernel(Q)` and
`MixtureMatrixRBFKernel(Q)` use the matrix-valued kernels of Wang et al.
(2019), where `Q` is a symmetric positive-definite matrix or a function that
returns one for a particle. By default, the bandwidth is recalculated at each
iteration with the median heuristic. Set `bandwidth` to a positive number to
use a fixed bandwidth.

The particle updates use AdaGrad with decay `α` and step size `stepsize`.

# Arguments

- `lp`: target implementing the first-order `LogDensityProblems.jl` interface
- `log_p`: function returning the possibly unnormalized log density at a particle
- `∇log_p`: function returning the gradient of `log_p` at a particle
- `inits`: matrix whose rows contain the initial particle positions
- `n_iter=100`: number of particle update iterations
- `stepsize=0.1`: particle update step size
- `kernel=RBFKernel()`: kernel used to calculate the SVGD direction
- `α=0.9`: decay applied to the AdaGrad history of squared `phi` values
- `bandwidth=nothing`: RBF kernel bandwidth; `nothing` applies the median
  heuristic at each iteration
- `show_progress=true`: display progress bar

The kernel constructors are:

- `RBFKernel()`: scalar RBF kernel; no arguments
- `AverageMatrixRBFKernel(Q)`: `Q` is a fixed symmetric positive-definite
  matrix or a function returning one matrix for a particle; function results
  are averaged
- `MixtureMatrixRBFKernel(Q)`: `Q` has the same forms and supplies the geometry
  for each current-particle mixture anchor

# Return value

A named tuple with fields:

- `particles`: a matrix whose rows are the final particles
- `log_p`: the log density at each final particle

# Reference

Liu, Q. and Wang, D. (2016). Stein Variational Gradient Descent: A General
Purpose Bayesian Inference Algorithm. Advances in Neural Information
Processing Systems 29.

Wang, D., Tang, Z., Bajaj, C. and Liu, Q. (2019). Stein Variational Gradient
Descent with Matrix-Valued Kernels. Advances in Neural Information Processing
Systems 32.
"""
function svgd(lp, inits::AbstractMatrix; n_iter=100, stepsize=0.1,
              kernel=RBFKernel(), bandwidth=nothing, α=0.9,
              show_progress=true)

    d = LogDensityProblems.dimension(lp)
    n = size(inits, 1)              # number of particles
    fudge_factor = 1e-6             # for adagrad update

    LogDensityProblems.capabilities(lp) >= LogDensityProblems.LogDensityOrder(1) ||
        error("The LogDensityProblem must provide gradient computation!")
    size(inits, 2) == d ||
        error("The initial positions must be given by an n x $(d) -matrix!")

    pos = float.(inits)         # initial particle positions
    ∇log_p = similar(pos)
    log_p = similar(pos, n)
    phi2_prev = zeros(eltype(pos), size(pos))

    progress = ProgressMeter.Progress(n_iter; dt=1, desc="Optimizing... ",
                                      enabled=show_progress)
    for l in 1:n_iter
        for i in 1:n
            _, ∇log_p_i = LogDensityProblems.logdensity_and_gradient(lp, @view pos[i, :])
            ∇log_p[i, :] .= ∇log_p_i
        end

        phi = stein_direction(kernel, pos, ∇log_p, bandwidth)

        # The first AdaGrad accumulator matches the reference implementation:
        # https://github.com/DartML/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py
        if l == 1
            phi2_prev .= phi.^2
        else
            phi2_prev .= α .* phi2_prev .+ (1 - α) .* phi.^2
        end
        adjusted_grad = phi ./ (fudge_factor .+ sqrt.(phi2_prev))
        pos .+= stepsize .* adjusted_grad

        ProgressMeter.next!(progress)
    end

    for i in 1:n
        log_p[i] = LogDensityProblems.logdensity(lp, @view pos[i, :])
    end
    (particles=pos, log_p=log_p)
end

function svgd(log_p::Function, ∇log_p::Function, inits::AbstractMatrix;
              kwargs...)
    lp = SimpleLogDensityProblem(log_p, ∇log_p, size(inits, 2))
    svgd(lp, inits; kwargs...)
end

end
