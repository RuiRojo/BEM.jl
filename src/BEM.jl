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
import IterativeSolvers
import Memoization
using Memoization: @memoize
using NearestNeighbors: KDTree, inrange
using Dates
using TimerOutputs

using MKL
# using Distributed

export create_operator_accel, create_operator_exact_lm, create_operator_exact_matrix
export create_operator_exact_matrix_ref
export Versor, axis_sweep, BoundaryValues
export BoundaryCondition, Dirichlet, Neumann
export Formulation, DirectFormulation, IndirectFormulation
export target_strength, intensity, backscattering, forward_scattering
export scattering_boundary, scattering_farfield, scattering_nearfield
export scattering_nearfield_sphere
export gmres

export SurfaceIntegralOperator, L, M, Mt, N
export Exact, Accelerated, LinearMap

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




abstract type SurfaceIntegralOperator end
abstract type L <: SurfaceIntegralOperator end
abstract type M <: SurfaceIntegralOperator end
abstract type Mt <: SurfaceIntegralOperator end
abstract type N <: SurfaceIntegralOperator end
const S = const SingleLayer = L
const D = const DoubleLayer = M
const DnS = const DnSingleLayer = Mt
const DnD = const DnDoubleLayer = N

export S, D, DnS, DnD, L, M, Mt, N, SingleLayer, DoubleLayer, DnSingleLayer, DnDoubleLayer

const quad_default = @quad_gquts 6

include("general.jl")
include("boundary.jl")
include("lu-mkl-no-pivot.jl")
include("kernels.jl")
include("integral-operators/integral-operators.jl")
include("scattering.jl")
include("esfera-exacta.jl")


# to-do: check if in the published version I end up using these or not and delete accordingly (and generally all automatic differentiation stuff).
@inline grad(f::Function, x) = rgrad(f, x) + im * igrad(f, x)
@inline rgrad(f::Function, x) = ForwardDiff.gradient(real ∘ f, x)
@inline igrad(f::Function, x) = ForwardDiff.gradient(imag ∘ f, x)
@inline ddir(f, x, dir) = dot(dir, grad(f, x))
@inline der(f, x) = rder(f, x) + im * ider(f, x)
@inline rder(f, x) = ForwardDiff.derivative(real ∘ f, x)
@inline ider(f, x) = ForwardDiff.derivative(imag ∘ f, x)




for fun in (create_operator_accel, create_operator_exact_matrix), OP in (L, M, Mt, N)
    precompile(fun, (typeof(OP), LightMesh,))
end



include("api-helper.jl")
end
