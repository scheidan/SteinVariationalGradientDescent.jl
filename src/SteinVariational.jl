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
         α=0.9, fudge_factor=1e-6, show_progress=true)
```
or:
```julia
    svgd(log_p, ∇log_p, inits;
         n_iter=100, stepsize=0.1, bandwidth=nothing,
         α=0.9, fudge_factor=1e-6, show_progress=true)
```

The rows of `inits` are the initial particles. The target can be a
`LogDensityProblems.jl` object that provides first derivatives, or it can be
given as separate log-density and gradient functions. An RBF kernel is used;
by default, its bandwidth is recalculated at each iteration with the median
heuristic. Set `bandwidth` to a positive number to use a fixed bandwidth.

The particle updates use AdaGrad with decay `α` and initial step size
`stepsize`, as in the reference implementation accompanying Liu and Wang
(2016).

# Arguments

- `lp`: target implementing the first-order `LogDensityProblems.jl` interface
- `log_p`: function returning the possibly unnormalized log density at a particle
- `∇log_p`: function returning the gradient of `log_p` at a particle
- `inits`: matrix whose rows contain the initial particle postions
- `n_iter=100`: number of particle update iterations
- `stepsize=0.1`: initial AdaGrad step size
- `α=0.9`: decay applied to the AdaGrad history of squared update directions
- `bandwidth=nothing`: RBF kernel bandwidth; `nothing` applies the median
  heuristic at each iteration
- `fudge_factor=1e-6`: value added to the AdaGrad denominator for numerical
  stability
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
              bandwidth=nothing, α=0.9, fudge_factor=1e-6,
              show_progress=true)
    particles = float.(inits)
    scores = similar(particles)
    log_ps = similar(particles, size(particles, 1))
    historical_gradient = zeros(eltype(particles), size(particles))

    progress = ProgressMeter.Progress(n_iter; dt=1, desc="Optimizing... ",
                                      enabled=show_progress)
    for iteration in 1:n_iter
        evaluate_target!(scores, log_ps, lp, particles)
        direction = svgd_direction(particles, scores, bandwidth)
        if iteration == 1
            historical_gradient .= direction.^2
        else
            historical_gradient .= α .* historical_gradient .+
                (1 - α) .* direction.^2
        end
        particles .+= stepsize .* direction ./
            (sqrt.(historical_gradient) .+ fudge_factor)
        ProgressMeter.next!(progress)
    end

    evaluate_target!(scores, log_ps, lp, particles)
    (particles=particles, log_p=log_ps)
end

function svgd(log_p::Function, ∇log_p::Function, inits::AbstractMatrix;
              kwargs...)
    lp = SimpleLogDensityProblem(log_p, ∇log_p, size(inits, 2))
    svgd(lp, inits; kwargs...)
end

function evaluate_target!(scores, log_ps, lp, particles)
    for i in axes(particles, 1)
        log_ps[i], gradient = LogDensityProblems.logdensity_and_gradient(
            lp, @view particles[i, :])
        scores[i, :] .= gradient
    end
end

function svgd_direction(particles, scores, bandwidth)
    n_particles, dimension = size(particles)
    squared_distances = zeros(eltype(particles), n_particles, n_particles)
    pairwise_distances = Vector{eltype(particles)}(
        undef, n_particles * (n_particles - 1) ÷ 2)

    index = 1
    for j in 1:n_particles-1, i in j+1:n_particles
        distance = zero(eltype(particles))
        for k in 1:dimension
            distance += abs2(particles[i, k] - particles[j, k])
        end
        squared_distances[i, j] = squared_distances[j, i] = distance
        pairwise_distances[index] = distance
        index += 1
    end

    h = isnothing(bandwidth) ? median_bandwidth(pairwise_distances,
                                                n_particles) : bandwidth
    h = max(h, eps(eltype(particles)))
    direction = zeros(eltype(particles), size(particles))
    for i in 1:n_particles, j in 1:n_particles
        kernel = exp(-squared_distances[i, j] / h)
        for k in 1:dimension
            direction[i, k] += kernel * scores[j, k] +
                2 * kernel * (particles[i, k] - particles[j, k]) / h
        end
    end
    direction ./ n_particles
end

function median_bandwidth(pairwise_distances, n_particles)
    if isempty(pairwise_distances)
        return one(eltype(pairwise_distances))
    end
    median(pairwise_distances) / log(eltype(pairwise_distances)(n_particles + 1))
end

end
