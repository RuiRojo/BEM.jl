using MKL_jll, LinearAlgebra

import LinearAlgebra: BlasInt


function lu_mkl_nopivot!(r::Matrix{Float64})
    n = size(r, 1)
    ccall(
        ("mkl_dgetrfnp", libmkl_rt), 
        Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt}),
        n,             n,             r,           n,            Ref{BlasInt}())
    return UnitLowerTriangular(r), UpperTriangular(r)
end
function lu_mkl_nopivot!(r::Matrix{ComplexF64})
    n = size(r, 1)
    ccall(
        ("mkl_zgetrfnp", libmkl_rt), 
        Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ptr{ComplexF64}, Ref{BlasInt}, Ptr{BlasInt}),
        n,             n,             r,           n,            Ref{BlasInt}())
    return UnitLowerTriangular(r), UpperTriangular(r)
end
function lu_mkl_nopivot!(r::Matrix{Float32})
    n = size(r, 1)
    ccall(
        ("mkl_sgetrfnp", libmkl_rt), 
        Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ptr{Float32}, Ref{BlasInt}, Ptr{BlasInt}),
        n,             n,             r,           n,            Ref{BlasInt}())
    return UnitLowerTriangular(r), UpperTriangular(r)
end
function lu_mkl_nopivot!(r::Matrix{ComplexF32})
    n = size(r, 1)
    ccall(
        ("mkl_cgetrfnp", libmkl_rt), 
        Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ptr{ComplexF32}, Ref{BlasInt}, Ptr{BlasInt}),
        n,             n,             r,           n,            Ref{BlasInt}())
    return UnitLowerTriangular(r), UpperTriangular(r)
end
lu_mkl_nopivot(r) = lu_mkl_nopivot!(copy(r))

