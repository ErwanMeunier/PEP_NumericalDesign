# PEPDesign — parametric Performance Estimation and step-size policy design.
#
# Pipeline: DSL (gram.jl) → exact degree-2 extraction (compile.jl) → SDP solve
# (solve.jl) → envelope gradients / frozen-certificate curvature (sensitivity.jl)
# → policy pullback and design optimizers (design/).

module PEPDesign

using JuMP
using Mosek
using MosekTools
using LinearAlgebra
using SparseArrays
using ForwardDiff
using Printf
using Random
using Statistics

include("params.jl")
include("gram.jl")
include("compile.jl")
include("solve.jl")
include("sensitivity.jl")
include("classes.jl")
include("frontend/core.jl")
include("frontend/functions.jl")
include("frontend/operators.jl")
include("frontend/steps.jl")
include("policies.jl")
include("design/oracle.jl")
include("design/trace.jl")
include("design/fom.jl")
include("design/som.jl")
include("design/ssdp.jl")
include("design/hrdp.jl")
include("methods/ogd.jl")
include("methods/item.jl")
include("methods/igdm.jl")

# Symbolic layer
export PAff, PQuad, coeff, affmul, quadmul, evaluate
# DSL
export PEPModel, PointExpr, FVal, QExpr,
    point!, fval!, coeffs!, add_le!, add_eq!, add_psd!,
    objective_max!, objective_maxmin!, inner, sqnorm
# Compilation
export CompiledPEP, ParamMatrix, CompiledPSD, compile, assemble, dmat, d2mat, trprod
# Solving
export SDPBackend, MosekBackend, GenericBackend, PEPSolution, solve_pep
# Sensitivity
export grad_hess_eta, pullback
# Frontend: oracles and combinations (ported from PEPit.jl, MIT)
export AbstractPEPFunction, PEPFunc, FunLinComb,
    oracle!, gradient!, value!, stationary_point!, fixed_point!,
    add_oracle_point!, model_of, npoints, triples, adjoint_oracle!
# Frontend: function classes
export ConvexFunction, StronglyConvexFunction, SmoothFunction,
    SmoothConvexFunction, SmoothStronglyConvexFunction,
    ConvexLipschitzFunction, SmoothConvexLipschitzFunction,
    ConvexIndicatorFunction, ConvexSupportFunction, ConvexQGFunction,
    RsiEbFunction, SmoothStronglyConvexQuadraticFunction,
    SmoothQuadraticLojasiewiczFunctionCheap,
    SmoothQuadraticLojasiewiczFunctionExpensive
# Frontend: operator classes
export MonotoneOperator, StronglyMonotoneOperator, LipschitzOperator,
    CocoerciveOperator, NonexpansiveOperator, NegativelyComonotoneOperator,
    LipschitzStronglyMonotoneOperatorCheap,
    LipschitzStronglyMonotoneOperatorExpensive,
    CocoerciveStronglyMonotoneOperatorCheap,
    CocoerciveStronglyMonotoneOperatorExpensive,
    LinearOperator, SymmetricLinearOperator, SkewSymmetricLinearOperator
# Frontend: primitive steps
export proximal_step!, inexact_proximal_step!, inexact_gradient!,
    inexact_gradient_step!, exact_linesearch_step!, linear_optimization_step!,
    shifted_optimization_step!, bregman_gradient_step!, bregman_proximal_step!,
    epsilon_subgradient_step!
# Policies
export AbstractPolicy, IdentityPolicy, ConstantPolicy, FunctionPolicy,
    PowerLawPolicy, ProductPolicy, MappedPolicy, evaluate_policy, nparams,
    policy_label,
    sum_of_exp_policy, log_poly_policy, dct_policy, cosine_decay_policy,
    rational_policy, warped_chebyshev_policy, piecewise_exp_policy
# Design
export DesignProblem, pep_value, pep_grad_hess, eval_all,
    AbstractDesignObjective,
    DesignTrace, best_point, design_fom, design_som,
    SSDPTrace, design_ssdp,
    HRDPObjective, SampledHRDP, snw, wgc, generalization_ratio, compute_wstar
# Methods
export ogd_pep, compile_ogd,
    item_pep, compile_item, item_optimal_params,
    igdm_pep, compile_igdm, igdm_flat_index, igdm_dof, igdm_diag_indices,
    igdm_hmem, igdm_diagonal_policy,
    IGDMKMemoryPolicy, IGDMStationaryPolicy, IGDMLagPolicy,
    add_interpolation_fmuL!

end # module
