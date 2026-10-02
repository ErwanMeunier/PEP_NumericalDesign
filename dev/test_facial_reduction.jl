# Standalone wrapper used by dev/run_all_tests.jl.
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

include(joinpath(@__DIR__, "..", "src", "PEPDesign.jl"))
using .PEPDesign
using Test

include(joinpath(@__DIR__, "..", "test", "facial_reduction.jl"))
println("test_facial_reduction: OK")
