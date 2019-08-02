module SimpleSVMs

import StatsBase
using JuMP
using LinearAlgebra: ⋅

struct SVM{W, B}
    w::W
    b::B
    SVM(w::W, b::B) where {W, B} = new{W,B}(w,b)
end

function StatsBase.fit(::Type{<:SVM}, penalty, X, y, optimizer)
    (m, w, b) = build_svm(penalty, X, y, optimizer)
    optimize!(m)
    @assert termination_status(m) == JuMP.MOI.OPTIMAL
    return SVM(JuMP.value.(w), JuP.value.(b))
end

function build_svm(penalty, X, y, optimizer)
    (nobs, nfeat) = size(X)
    length(y) == nobs || throw(DimensionMismatch("X and y need the same number of observations"))
    m = JuMP.Model(optimizer)
    @variable(m, w[1:nfeat])
    @variable(m, b)
    @variable(m, l[1:nobs] ≥ 0)
    @constraint(m, hinge_loss[i=1:nobs], l[i] ≥ 1 - y[i] * (X[i,:]⋅w + b))
    restrict_weights(m, penalty, w)
    @objective(m, Min, sum(l))
    return (m, w, b)
end

struct L1Penalty{R}
    rhs::R
    L1Penalty(rhs::R) where {R} = new{R}(rhs)
end

function restrict_weights(m::JuMP.AbstractModel, l1::L1Penalty, w)
    nfeat = length(w)
    @variable(m, wm[1:nfeat] ≥ 0)
    @variable(m, wp[1:nfeat] ≥  0)
    @constraint(m, sum(wm) + sum(wp) ≤ l1.rhs)
end

struct L2Penalty{R}
    rhs::R
    L2Penalty(rhs::R) where {R} = new{R}(rhs)
end

function restrict_weights(m::JuMP.AbstractModel, l2::L2Penalty, w)
    @constraint(m, w ⋅ w ≤ l2.rhs)
end

end # module
