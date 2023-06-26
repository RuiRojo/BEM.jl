module BEM


using Reexport
@reexport using Mesh
using IterTools, LinearAlgebra, Statistics, FFTW
using StaticArrays, SparseArrays#, NearestNeighbors
using ThreadsX#, Transducers
using LinearMaps 
using ForwardDiff: gradient
import ForwardDiff
# import ProgressMeter # Tmp: replace by ProgressLogging
using ProgressLogging: @progress, @withprogress, @logprogress
using IterativeSolvers: gmres, gmres!
import Memoization
using Memoization: @memoize
using NearestNeighbors: KDTree, inrange
using Dates
using TimerOutputs

using MKL
# using Distributed

export create_operator_approx, create_operator_exact_lm, create_operator_exact_matrix
export create_operator_exact_matrix_lavia
export Versor, axis_sweep, BoundaryValues
export BoundaryCondition, Dirichlet, Neumann
export Formulation, DirectFormulation, IndirectFormulation
export target_strength, intensity, backscattering, forward_scattering
export scattering_boundary, scattering_farfield, scattering_nearfield
export scattering_nearfield_sphere

function __init__()
    # FFTW.set_num_threads(Threads.nthreads())
    FFTW.set_provider!("mkl")
    FFTW.set_num_threads(Sys.CPU_THREADS)    
    # BLAS.set_num_threads(Threads.nthreads())
end

const CACHE = Ref(false) #true

const RUN_LOG = Dict()

const to = TimerOutput()

include("macros.jl")

const MAT_EL_TYPE = ComplexF32
const Float = Float32 # ¿El de los BoundaryValues nada más?




abstract type Operator end
abstract type L <: Operator end
abstract type M <: Operator end
abstract type Mt <: Operator end
abstract type N <: Operator end
const S = const SingleLayer = L
const D = const DoubleLayer = M
const DnS = const DnSingleLayer = Mt
const DnD = const DnDoubleLayer = N

export S, D, DnS, DnD, L, M, Mt, N, SingleLayer, DoubleLayer, DnSingleLayer, DnDoubleLayer

const quad_default = @quad_gquts 6

include("general.jl")
include("boundary.jl")
include("lu-mkl-no-pivot.jl")
include("rui-integrals.jl")
include("exact.jl")
include("scattering.jl")
include("op_edm.jl")
include("accelerated.jl")
include("esfera-exacta.jl")

@inline grad(f::Function, x) = rgrad(f, x) + im * igrad(f, x)
@inline rgrad(f::Function, x) = ForwardDiff.gradient(real ∘ f, x)
@inline igrad(f::Function, x) = ForwardDiff.gradient(imag ∘ f, x)
@inline ddir(f, x, dir) = dot(dir, grad(f, x))
@inline der(f, x) = rder(f, x) + im * ider(f, x)
@inline rder(f, x) = ForwardDiff.derivative(real ∘ f, x)
@inline ider(f, x) = ForwardDiff.derivative(imag ∘ f, x)



#TMP



"""
    green(x, y; k, krmin=1e-3)
    green(xy_vec; k)
    green(r; k)

Return the acoustic field of a unit monopole source with wavenumber k.


exp(im * k * r) / (4π * r)

where r == norm(xy_vec) and xy_vec == x - y.

When k * r is very small, it returns zero.
"""
function green(r; k, krmin=1e-9)
    kr = k * r
    kr > krmin && return cis(kr) / (4 * r * π)
    return (im * k * cis(k / 2 * r) * sinc((k / 2) / π * r) + 1 / r) / 4 / π
end
green(x, y; k, krmin=1e-10) = green(norm(x .- y); k, krmin)
green(x::AbstractVector; kwargs...) = green(norm(x); kwargs...)





for fun in (create_operator_approx, create_operator_exact_matrix), OP in (L, M, Mt, N)
    precompile(fun, (typeof(OP), LightMesh,))
end

#=
let
    vert = [rand(SVector{3, Float32}) for _ in 1:1000]
    selv = [ (rand(1:1000), rand(1:1000), rand(1:1000)) for _ in 1:5]
    mesh = LightMesh(vert, selv)  

    create_s_operator(mesh; k=2.3, parallel_tasks=false)
end
=#


include("bs-params.jl")
end
