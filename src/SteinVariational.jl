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
    particles = float.(inits)
    scores = similar(particles)
    log_ps = similar(particles, size(particles, 1))
    phi_previous = zeros(eltype(particles), size(particles))

    progress = ProgressMeter.Progress(n_iter; dt=1, desc="Optimizing... ",
                                      enabled=show_progress)
    for l in 1:n_iter

        # get gradient of each particle
        for p in axes(particles, 1)
            log_ps[p], gradient = LogDensityProblems.logdensity_and_gradient(lp, @view particles[p, :])
            scores[p, :] .= gradient
        end


        phi = SteinVariational.phi(particles, scores, bandwidth)
        phi .= α .* phi_previous .+ (1 - α) .* phi
        phi_previous .= phi
        particles .+= stepsize .* phi

        ProgressMeter.next!(progress)
    end

    for p in axes(particles, 1)
        log_ps[p] = LogDensityProblems.logdensity(lp, @view particles[p, :])
    end
    (particles=particles, log_p=log_ps)
end

function svgd(log_p::Function, ∇log_p::Function, inits::AbstractMatrix;
              kwargs...)
    lp = SimpleLogDensityProblem(log_p, ∇log_p, size(inits, 2))
    svgd(lp, inits; kwargs...)
end




function phi(particles, scores, bandwidth)
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

    phi = zeros(eltype(particles), size(particles))
    for i in 1:n_particles, j in 1:n_particles
        kernel = exp(-squared_distances[i, j] / h)
        for k in 1:dimension
            phi[i, k] += kernel * scores[j, k] +
                2 * kernel * (particles[i, k] - particles[j, k]) / h
        end
    end
    phi ./ n_particles
end

function median_bandwidth(pairwise_distances, n_particles)
    if isempty(pairwise_distances)
        return one(eltype(pairwise_distances))
    end
    median(pairwise_distances) / log(eltype(pairwise_distances)(n_particles + 1))
end

end
