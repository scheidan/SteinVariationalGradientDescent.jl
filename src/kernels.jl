function pairwise_squared_distances(x)
    squared_norms = sum(abs2, x; dims=2)
    max.(squared_norms .+ squared_norms' .- 2 .* (x * x'), zero(eltype(x)))
end

function pairwise_squared_distances(x, Q)
    xQ = x * Q
    squared_norms = sum(xQ .* x; dims=2)
    max.(squared_norms .+ squared_norms' .- 2 .* (xQ * x'), zero(eltype(xQ)))
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

function kernel_bandwidth(squared_distances, bandwidth)
    h = isnothing(bandwidth) ? median_bandwidth(squared_distances) : bandwidth
    max(h, eps(eltype(squared_distances)))
end

geometry_matrices(Q::AbstractMatrix, pos) = fill(Q, size(pos, 1))
geometry_matrices(Q, pos) = [Q(@view pos[i, :]) for i in axes(pos, 1)]

# Particle vectors are stored as rows, so this applies Q^-1 on their right.
right_solve(factor, x) = transpose(factor \ transpose(x))

"""
    RBFKernel()

Scalar radial basis function kernel used by standard SVGD.
"""
struct RBFKernel end

function stein_direction(::RBFKernel, pos, score, bandwidth)
    n = size(pos, 1)
    squared_distances = pairwise_squared_distances(pos)
    h = kernel_bandwidth(squared_distances, bandwidth)
    k = exp.(-squared_distances ./ h)

    # The exponent is -distance^2/h, giving the factor 2/h in its derivative.
    repulsion = 2 / h .* (sum(k; dims=2) .* pos - k * pos)
    (k * score + repulsion) ./ n
end

"""
    AverageMatrixRBFKernel(Q)

Matrix-valued RBF kernel using a particle-averaged geometry. `Q` can be a
symmetric positive-definite matrix or a function `x -> Q(x)`.
"""
struct AverageMatrixRBFKernel{T}
    Q::T
end

average_geometry(Q::AbstractMatrix, pos) = Q
function average_geometry(Q, pos)
    sum(geometry_matrices(Q, pos)) / size(pos, 1)
end

function stein_direction(kernel::AverageMatrixRBFKernel, pos, score, bandwidth)
    n = size(pos, 1)
    Q = average_geometry(kernel.Q, pos)
    factor = cholesky(Q)
    squared_distances = pairwise_squared_distances(pos, Q)
    h = kernel_bandwidth(squared_distances, bandwidth)
    k = exp.(-squared_distances ./ h)

    # Q is held fixed for this update, so Q^-1 cancels Q in the kernel derivative.
    attraction = right_solve(factor, k * score)
    repulsion = 2 / h .* (sum(k; dims=2) .* pos - k * pos)
    (attraction + repulsion) ./ n
end

"""
    MixtureMatrixRBFKernel(Q)

Mixture of matrix-valued RBF kernels centered on the current particles. `Q`
can be a symmetric positive-definite matrix or a function `x -> Q(x)`.
"""
struct MixtureMatrixRBFKernel{T}
    Q::T
end

function stein_direction(kernel::MixtureMatrixRBFKernel, pos, score, bandwidth)
    n = size(pos, 1)
    # Each particle is an anchor; its Q is fixed while differentiating the kernel.
    Qs = geometry_matrices(kernel.Q, pos)
    factors = cholesky.(Qs)
    component_scores = [-(pos .- @view(pos[l:l, :])) * Qs[l] for l in 1:n]
    T = promote_type(eltype(pos), eltype(first(component_scores)))
    log_weights = similar(pos, T, n, n)

    for l in 1:n
        differences = pos .- @view(pos[l:l, :])
        squared_distances = vec(sum((-component_scores[l]) .* differences; dims=2))
        log_weights[l, :] .= logdet(factors[l]) / 2 .- squared_distances / 2
    end

    # Column-wise normalization gives each particle's weights across anchors.
    log_weights .-= maximum(log_weights; dims=1)
    weights = exp.(log_weights)
    weights ./= sum(weights; dims=1)

    average_component_score = similar(pos, T, size(pos))
    fill!(average_component_score, zero(T))
    for l in 1:n
        average_component_score .+= @view(weights[l, :]) .* component_scores[l]
    end

    phi = similar(pos, T, size(pos))
    fill!(phi, zero(T))
    for l in 1:n
        Q = Qs[l]
        squared_distances = pairwise_squared_distances(pos, Q)
        h = kernel_bandwidth(squared_distances, bandwidth)
        k = exp.(-squared_distances ./ h)
        w = @view weights[l, :]
        ∇log_w = component_scores[l] - average_component_score

        # Differentiating w_l(x) adds ∇log(w_l) to the source-particle score.
        attraction = right_solve(factors[l], k * (w .* (score + ∇log_w)))
        weighted_kernel_sum = k * w
        repulsion = 2 / h .* (weighted_kernel_sum .* pos - k * (w .* pos))
        phi .+= w .* (attraction + repulsion)
    end
    phi ./ n
end
