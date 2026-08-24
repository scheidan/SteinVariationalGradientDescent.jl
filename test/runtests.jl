using SteinVariational
using Statistics
using Test

log_p_standard_normal(x) = -sum(abs2, x) / 2
∇log_p_standard_normal(x) = -x

@testset "SteinVariational" begin
    @testset "SVGD phi" begin
        particles = reshape([-1.0, 1.0], 2, 1)
        scores = -particles
        kernel = exp(-2)
        expected = reshape([(1 - 3kernel) / 2, -(1 - 3kernel) / 2], 2, 1)
        phi = SteinVariational.phi(particles, scores, 2.0)
        @test phi ≈ expected
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
        lp = SteinVariational.SimpleLogDensityProblem(
            log_p_standard_normal, ∇log_p_standard_normal, 2)
        result_lp = svgd(lp, inits; n_iter=5, show_progress=false)
        result_functions = svgd(log_p_standard_normal,
                                ∇log_p_standard_normal, inits;
                                n_iter=5, show_progress=false)
        @test result_lp == result_functions
    end
end
