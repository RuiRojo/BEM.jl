#=function outer(fun :: Function, xs, ys)
    [ fun(x, y) for x in xs, y in ys]
end

flatten(x::Vector{<:Vector}) = reshape(vcat(x...), size(first(x), 1), size(x, 1))
unflatten(x::Matrix) = [ x[:,i] for i in 1:size(x, 2) ]
unflatten(x::Vector{<:Vector}) = x
=#



struct Versor <: AbstractVector{Float64}
    φ :: Float64
    θ :: Float64
end
Versor(x, y, z) = Versor(
    atan(y, x),
    acos(z / norm((x, y, z))),
    )
Versor(x::AbstractVector) = Versor(x...)

import Base.-
Base.:-(v::Versor) = Versor(-SVector(v))
Base.size(::Versor) = (3, )
function Base.getindex(v::Versor, i::Integer)
    i == 1 && return sin(v.θ) * cos(v.φ)
    i == 2 && return sin(v.θ) * sin(v.φ)
    i == 3 && return cos(v.θ)
    throw("Bad index")
end
StaticArrays.SVector(v::Versor) = SVector(v[1], v[2], v[3])


facet_vs_λ(mesh; k) = (2π / k) / edgelenstats(mesh).deciles[9]

maxfreq(mesh) = 1500 / (edgelenstats(mesh).deciles[9] * 6)