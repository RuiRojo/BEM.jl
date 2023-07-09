
"""
h(ξs; ns=$(_def_h_nterms))

Return the interpolating function from neighbouring points ξs.
"""
function h(ξs; ns, scale)
    offset = mean(ξs)
    ξs_offset = map(x -> Float32.(x - offset), ξs)

    k = scale ? maximum(abs.(ξs_offset[2] - ξs_offset[1])) : 1
    #scale = mean(norm.(ξs_offset)) # Test    
    ξs_scaled = ξs_offset ./ k # Test

    len = length(ξs_offset)
    n = ns[findlast(<=(len), ns)] # Assumes ns is sorted
    #y -> _h(ξs_offset; n)(y - offset)
    y -> _h(ξs_scaled; n)((y - offset) / k) # Test
end
function _h end
push!(MEMO_FUNS, _h)
@memoize Dict function _h(ξs_off; n)
    F = [ f(Float32.(ξi)) for f in fs(n), ξi in vec(ξs_off) ]
    Finv = SMatrix{length(ξs_off), n}(pinv(F))
    fsN = SMatrix{n, 1}(fs(n))

    function h(y)
        # f = vec([ f(y) for f in fs(n) ])
        f = map(f -> f(y), fsN)
        #f = [ f(y) for f in fs(n) ]

        #return vec(pinv(F#=; rtol=sqrt(eps(real(float(one(eltype(F))))))=#) * f)
        return vec(Finv * f) 
        # To-do: I think solving the linear system should be faster 
        # Also: is this allocating with the `vec`?
    end

    return h
end

"""
f_coefs(n)

Return a vector with the coefficients of the source coordinates ξ to be used 
in the approximation of the Green function.

Each coefficient is given as a SVector{3, Int} with the power of each coordinate.

# Example

```jldoctest
julia> f_coefs(1)

julia> f_coefs(8)

julia> f_coefs(17)
```
"""
function f_coefs(n)
    function lt(x, y)
        s = sign(sum(x) - sum(y)) 
        !iszero(s) && return s < 0

        mp = sign(maximum(x) - maximum(y))
        !iszero(mp) && return mp < 0

        return x < y
    end

    candidates = let 
        imax = ceil(Int, sqrt(n)) # prov
        [ Iterators.product(0:imax, 0:imax, 0:imax)... ] 
    end

    return sort(candidates; lt)[1:n]
end


#=
"Does f_coefs(n) produce a basis that is symmetric in the 3 spacial coordinates?"
function issym_interp_coef(n)
    perms = [ [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    t1 = f_coefs(n)

    sym = all(perms) do p
        sort(t1) == sort(map(t -> t[p], t1))
    end
end
=#


interp_coefs_upto(n) = filter(<=(n), [ 8, 17, 32, 53, 81 ])#[ i for i in 8:n if issym_interp_coef(i) ]





"Return the vector of functions that take the source ξ to give the f_i coefficient"
push!(MEMO_FUNS, fs)
@memoize fs(n) = [ ξ -> sum(ξ .^ f_coef) for f_coef in f_coefs(n) ]


## Interpolator

"Functor that takes the grid field values and returns the field value approximation on the mesh facets"
struct Interpolator{T}
    mesh :: LightMesh
    grid      :: Grid
    neighsidx :: Vector{NTuple{3, UnitRange{Int64}}}
    hvecs     :: Vector{Vector{Float32}}
    hfuns     :: Vector{T}
end

function Interpolator(mesh; gridfactor=_def_gridfactor, verbose=false, kwargs...)
    grid = Grid(mesh; gridfactor)
    verbose && @myinfo "Grid size: $(size(grid))"
    Interpolator(mesh, grid; verbose, kwargs...)
end

push!(MEMO_FUNS, Interpolator)
@memoize Interpolator(mesh, grid; ns=_def_h_nterms, scale, verbose=false) = mytime("Interpolator"; verbose) do

    # The centroids of each facet
    xcollocation = mean.(mesh)

    neighsidx = get_facet_neighbourhoods(mesh, grid)

    hfuns = map(ξs -> h(ξs; scale, ns), get_neighbouring_nodes(grid, neighsidx))

    hvecs = [ Vector(hf(ξ)) for (hf, ξ) in zip(hfuns, xcollocation) ]


    return Interpolator(mesh, grid, neighsidx, hvecs, hfuns)
end


Base.:*(int::Interpolator, φ_grid) = @mytime "Interpolating" begin
    hvecs = int.hvecs
    neighsidx = int.neighsidx

    φs = map(hvecs, neighsidx) do h, ineigh
        dot(h, φ_grid[CartesianIndices(ineigh)])
    end

    return φs
end

Base.:*(int::Interpolator, φ_grids::AbstractMatrix) = int * sum(φ_grids)
function Base.:*(int::Interpolator, φ_grids::SMatrix{<:Any, 3})
    normal_components = eachrow(reinterpret(reshape, Float32, normal.(int.mesh)))
    return sum(n .* (int * φ_grid) for (n, φ_grid) in zip(normal_components, sum(φ_grids; dims=1)))
end



"Return the coordinates of the neighbouring nodes as a vector (an element per face) of 3D arrays"
get_neighbouring_nodes(interp::Interpolator) = get_neighbouring_nodes(interp.grid, interp.neighsidx)
get_neighbouring_nodes(grid, neighsidx) = map(nei -> grid[nei...], neighsidx)


