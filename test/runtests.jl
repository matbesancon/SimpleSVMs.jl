using SimpleSVMs
using Test
import Clp
using JuMP
import Random
import Ipopt

@testset "Improving accuracy with relaxed w" begin
    Random.seed!(33)
    X = vcat(rand(20, 2), rand(30,2) .+ [1,0.5]')
    y = append!(ones(20), -ones(30))
    prevloss = 500.0
    for rhs in [0.0, 0.1, 0.5]
        (m, w, b) = SimpleSVMs.build_svm(SimpleSVMs.L1Penalty(rhs), X, y, with_optimizer(Clp.Optimizer, LogLevel = 0))
        optimize!(m)
        res = JuMP.objective_value(m)
        @test res < prevloss
        prevloss = res
        if res ≈ 0.0
            break
        end
    end
    @test prevloss ≈ 0
end

@testset "Non-separable domains" begin
    Random.seed!(33)
    X = vcat(randn(20, 2), randn(30,2) .+ [3.0,1.5]')
    y = append!(ones(20), -ones(30))
    prevloss = 500.0
    for rhs in [0.0, 0.1, 0.5, 1.5, 100.0]
        (m, w, b) = SimpleSVMs.build_svm(SimpleSVMs.L1Penalty(rhs), X, y, with_optimizer(Clp.Optimizer, LogLevel = 0))
        optimize!(m)
        res = JuMP.objective_value(m)
        @test res ≤ prevloss
        prevloss = res
    end
    @test prevloss > 0
end

@testset "L2-restriction" begin
    Random.seed!(33)
    X = vcat(randn(20, 2), randn(30,2) .+ [3.0,1.5]')
    y = append!(ones(20), -ones(30))
    prevloss = 500.0
    for rhs in [0.0, 0.1, 0.5, 1.5, 100.0]
        (m, w, b) = SimpleSVMs.build_svm(SimpleSVMs.L2Penalty(rhs), X, y, with_optimizer(Ipopt.Optimizer, print_level = 0))
        optimize!(m)
        res = JuMP.objective_value(m)
        @test res ≤ prevloss
        prevloss = res
    end
    @test prevloss > 0
end
