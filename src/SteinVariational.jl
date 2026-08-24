module SteinVariational

export svgd

using Statistics: median
import LogDensityProblems
import ProgressMeter

include("logdensityproblem_interface.jl")

"""
Approximate a target distribution with Stein variational gradient descent.

```julia
    svgd(lp, inits;
         n_iter=100, stepsize=0.1, bandwidth=nothing,
         α=0.9, show_progress=true)
```
or:
```julia
    svgd(log_p, ∇log_p, inits;
         n_iter=100, stepsize=0.1, bandwidth=nothing,
         α=0.9, show_progress=true)
```

The rows of `inits` are the initial particles. The target can be a
`LogDensityProblems.jl` object that provides first derivatives, or it can be
given as separate log-density and gradient functions. An RBF kernel is used;
by default, its bandwidth is recalculated at each iteration with the median
heuristic. Set `bandwidth` to a positive number to use a fixed bandwidth.

The particle updates use exponential momentum with decay `α` and step size
`stepsize`.

# Arguments

- `lp`: target implementing the first-order `LogDensityProblems.jl` interface
- `log_p`: function returning the possibly unnormalized log density at a particle
- `∇log_p`: function returning the gradient of `log_p` at a particle
- `inits`: matrix whose rows contain the initial particle postions
- `n_iter=100`: number of particle update iterations
- `stepsize=0.1`: particle update step size
- `α=0.9`: weight assigned to the previous smoothed `phi`
- `bandwidth=nothing`: RBF kernel bandwidth; `nothing` applies the median
  heuristic at each iteration
- `show_progress=true`: display progress bar

# Return value

A named tuple with fields:

- `particles`: a matrix whose rows are the final particles
- `log_p`: the log density at each final particle

# Reference

Liu, Q. and Wang, D. (2016). Stein Variational Gradient Descent: A General
Purpose Bayesian Inference Algorithm. Advances in Neural Information
Processing Systems 29.
"""
function svgd(lp, inits::AbstractMatrix; n_iter=100, stepsize=0.1,
              bandwidth=nothing, α=0.9, show_progress=true)

    d = LogDensityProblems.dimension(lp)
    n = size(inits, 1)              # number of particles

    LogDensityProblems.capabilities(lp) >= LogDensityProblems.LogDensityOrder(1) ||
        error("The LogDensityProblem must provide gradient computation!")
    size(inits, 2) == d ||
    error("The initial positions must be given by an n x $(d) -matrix!")

    pos = float.(inits)         # initial particle positions
    ∇log_p = similar(pos)
    log_p = similar(pos, n)
    phi_previous = zeros(eltype(pos), size(pos))

    progress = ProgressMeter.Progress(n_iter; dt=1, desc="Optimizing... ",
                                      enabled=show_progress)
    for l in 1:n_iter
        for i in 1:n
            _, ∇log_p_i = LogDensityProblems.logdensity_and_gradient(lp, @view pos[i, :])
            ∇log_p[i, :] .= ∇log_p_i
        end

        # compute RBF kernel
        squared_distances = pairwise_squared_distances(pos)

        h = isnothing(bandwidth) ? median_bandwidth(squared_distances) : bandwidth
        h = max(h, eps(eltype(pos)))

        k = exp.(-squared_distances ./ h)

        # compute phi and update positions
        phi = (k * ∇log_p + 2 / h .* (sum(k; dims=2) .* pos - k * pos)) ./ n
        phi .= α .* phi_previous .+ (1 - α) .* phi
        phi_previous .= phi
        pos .+= stepsize .* phi

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




function pairwise_squared_distances(x)
    squared_norms = sum(abs2, x; dims=2)
    max.(squared_norms .+ squared_norms' .- 2 .* (x * x'), zero(eltype(x)))
end

function median_bandwidth(squared_distances)
    n = size(squared_distances, 1)
    if n < 2
        return one(eltype(squared_distances))
    end

    pairwise_distances = [squared_distances[i, j]
                          for j in 1:n-1 for i in j+1:n]
    median(pairwise_distances) /
        log(eltype(squared_distances)(n + 1))
end

end
