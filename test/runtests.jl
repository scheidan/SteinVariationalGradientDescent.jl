using SteinVariationalGradientDescent
using LinearAlgebra
using Statistics
using Test

log_p_standard_normal(x) = -sum(abs2, x) / 2
∇log_p_standard_normal(x) = -x

function direct_average_phi(pos, score, Q, h)
    n = size(pos, 1)
    phi = zeros(size(pos))
    for i in 1:n, j in 1:n
        difference = @view(pos[i, :]) - @view(pos[j, :])
        k = exp(-dot(difference, Q * difference) / h)
        phi[i, :] .+= k .* (Q \ @view(score[j, :])) .+
                      2 / h .* k .* difference
    end
    phi ./ n
end

function direct_mixture_phi(pos, score, Qs, h)
    n, d = size(pos)
    component_scores = zeros(n, n, d)
    log_weights = zeros(n, n)
    for l in 1:n, j in 1:n
        difference = @view(pos[j, :]) - @view(pos[l, :])
        Q_difference = Qs[l] * difference
        component_scores[l, j, :] .= -Q_difference
        log_weights[l, j] = logdet(Qs[l]) / 2 - dot(difference, Q_difference) / 2
    end
    log_weights .-= maximum(log_weights; dims=1)
    weights = exp.(log_weights)
    weights ./= sum(weights; dims=1)

    average_component_score = zeros(n, d)
    for j in 1:n, l in 1:n
        average_component_score[j, :] .+=
            weights[l, j] .* @view(component_scores[l, j, :])
    end

    phi = zeros(size(pos))
    for i in 1:n, j in 1:n, l in 1:n
        difference = @view(pos[i, :]) - @view(pos[j, :])
        Q_difference = Qs[l] * difference
        k = exp(-dot(difference, Q_difference) / h)
        ∇log_w = @view(component_scores[l, j, :]) -
                 @view(average_component_score[j, :])
        ∇k = 2 / h .* k .* Q_difference
        contribution = Qs[l] \ (k .* (@view(score[j, :]) + ∇log_w) + ∇k)
        phi[i, :] .+= weights[l, i] * weights[l, j] .* contribution
    end
    phi ./ n
end

@testset "SteinVariationalGradientDescent" begin
    @testset "SVGD update" begin
        particles = reshape([-1.0, 1.0], 2, 1)
        kernel = exp(-2)
        squared_distances = SteinVariationalGradientDescent.pairwise_squared_distances(particles)
        @test squared_distances == [0.0 4.0; 4.0 0.0]

        expected_phi = reshape([(1 - 3kernel) / 2,
                                -(1 - 3kernel) / 2], 2, 1)
        adjusted_phi = expected_phi ./ (1e-6 .+ sqrt.(expected_phi.^2))
        result = svgd(log_p_standard_normal, ∇log_p_standard_normal,
                      particles; n_iter=1, stepsize=1.0, bandwidth=2.0,
                      show_progress=false)
        @test result.particles ≈ particles + adjusted_phi
    end

    @testset "Function interface" begin
        inits = reshape(collect(range(-4.0, 6.0; length=40)), 20, 2)
        result = svgd(log_p_standard_normal, ∇log_p_standard_normal, inits;
                      n_iter=500, stepsize=0.05, show_progress=false)

        @test size(result.particles) == size(inits)
        @test result.log_p ≈ [log_p_standard_normal(x)
                              for x in eachrow(result.particles)]
        @test all(abs.(vec(mean(result.particles; dims=1))) .< 0.2)
        @test all(abs.(vec(std(result.particles; dims=1)) .- 1) .< 0.2)
    end

    @testset "LogDensityProblems interface" begin
        inits = reshape([-2.0, -1.0, 1.0, 2.0], 2, 2)
        lp = SteinVariationalGradientDescent.SimpleLogDensityProblem(
            log_p_standard_normal, ∇log_p_standard_normal, 2)
        result_lp = svgd(lp, inits; n_iter=5, show_progress=false)
        result_functions = svgd(log_p_standard_normal,
                                ∇log_p_standard_normal, inits;
                                n_iter=5, show_progress=false)
        @test result_lp == result_functions
    end

    @testset "Average matrix kernel" begin
        particles = [-1.0 0.5; 0.5 -1.0; 2.0 1.0]
        score = -particles
        Q = [4.0 0.2; 0.2 0.5]
        h = 1.7

        expected_phi = direct_average_phi(particles, score, Q, h)
        phi = SteinVariationalGradientDescent.stein_direction(
            AverageMatrixRBFKernel(Q), particles, score, h)
        @test phi ≈ expected_phi

        identity_kernel = AverageMatrixRBFKernel(Matrix{Float64}(I, 2, 2))
        for bandwidth in (nothing, h)
            scalar_phi = SteinVariationalGradientDescent.stein_direction(
                RBFKernel(), particles, score, bandwidth)
            matrix_phi = SteinVariationalGradientDescent.stein_direction(
                identity_kernel, particles, score, bandwidth)
            @test matrix_phi ≈ scalar_phi
        end

        callback_phi = SteinVariationalGradientDescent.stein_direction(
            AverageMatrixRBFKernel(_ -> Q), particles, score, h)
        @test callback_phi ≈ phi
    end

    @testset "Mixture matrix kernel" begin
        particles = [-1.0 0.5; 0.5 -1.0; 2.0 1.0]
        score = -particles
        Q(x) = [1.5 + x[1]^2 0.1; 0.1 1.0 + x[2]^2]
        Qs = [Q(x) for x in eachrow(particles)]
        h = 1.7

        expected_phi = direct_mixture_phi(particles, score, Qs, h)
        phi = SteinVariationalGradientDescent.stein_direction(
            MixtureMatrixRBFKernel(Q), particles, score, h)
        @test phi ≈ expected_phi

        fixed_Q = [2.0 0.1; 0.1 1.0]
        fixed_phi = SteinVariationalGradientDescent.stein_direction(
            MixtureMatrixRBFKernel(fixed_Q), particles, score, h)
        callback_phi = SteinVariationalGradientDescent.stein_direction(
            MixtureMatrixRBFKernel(_ -> fixed_Q), particles, score, h)
        @test callback_phi ≈ fixed_phi
        @test all(isfinite, SteinVariationalGradientDescent.stein_direction(
            MixtureMatrixRBFKernel(Q), particles, score, nothing))
    end

    @testset "Anisotropic normal" begin
        Q = [25.0 0.0; 0.0 0.25]
        log_p(x) = -dot(x, Q * x) / 2
        ∇log_p(x) = -(Q * x)
        grid1 = collect(range(-1.0, 1.0; length=7))
        grid2 = collect(range(-6.0, 6.0; length=7))
        inits = hcat(repeat(grid1; inner=7), repeat(grid2; outer=7))

        result = svgd(log_p, ∇log_p, inits;
                      kernel=AverageMatrixRBFKernel(Q), n_iter=200,
                      stepsize=0.05, show_progress=false)
        particle_mean = vec(mean(result.particles; dims=1))
        particle_std = vec(std(result.particles; dims=1))
        @test all(abs.(particle_mean) .< 0.05)
        @test particle_std ≈ [0.2, 2.0] atol=0.15
        @test result.log_p ≈ [log_p(x) for x in eachrow(result.particles)]
    end
end
