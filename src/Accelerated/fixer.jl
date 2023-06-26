#=
"""
Return the parsed `r_near`, which can be set through a combination of kwargs.

If `force_r_near` is false, then the returned value will not exceed a minimum safety margin.

Otherwise, return the safety margin unless either `λs_near` or `r_near` have been set.
"""
function parse_r_near(mesh, grid; λs_near, r_near, force_r_near) 

        # A hard minimum distance to force correction of nodes that "go to themselves"
    # To-do: note that one could pre-compute a stricter minimum than this one
    r_margin = 2edgelenstats(mesh).maximum + 2step(grid)

    @assert isnothing(λs_near) || isnothing(r_near)  "You can't set both `λs_near` and `r_near`"

    if isnothing(r_near)
        r_near = isnothing(λs_near) ? r_margin : λs_near * 2π / k
    end
    

    if !force_r_near && r_margin > r_near
        @warn "The distance between nearby facets was overriden from requested $r_near to $r_margin (not trimmed)."    
        r_near = max(r_near, r_margin)
    end

    return r_near
end
=#

function get_facet_neighbourhoods end
push!(MEMO_FUNS, get_facet_neighbourhoods)
@memoize #=@maybe_memoize=# IdDict get_facet_neighbourhoods(mesh, grid) = ThreadsX.map(f -> computeneighbourhood(f, grid), mesh)


function get_nearby_facets(mesh, grid; 
    λs_near=nothing, 
    r_near=nothing, 
    include_nearest=true,
    verbose=false,
    neigh_node_idxs=get_facet_neighbourhoods(mesh, grid)
    )
    
    @assert isnothing(λs_near) || isnothing(r_near)  "You can't set both `λs_near` and `r_near`"
    isnothing(r_near) && (r_near = isnothing(λs_near) ? 0 : λs_near * 2π / k)


    nearby_facets = _get_nearby_facets(
                        mesh, 
                        grid, 
                        r_near; 
                        verbose, 
                        include_nearest,
                        neigh_node_idxs
                    )

    verbose && let
        nflens = length.(nearby_facets)
        push!(RUN_LOG, "Nearby facet %" => 100sum(nflens)/length(mesh)^2)
        @myinfo "Nearby facets: $(sum(nflens)) -- avg per facet $(round(mean(nflens); digits=3)) -- $(round(RUN_LOG["Nearby facet %"]; digits=3))%"
    end

    return nearby_facets
end

function _get_nearby_facets end
push!(MEMO_FUNS, _get_nearby_facets)
@memoize IdDict function _get_nearby_facets(mesh, grid, r_near; include_nearest, verbose, neigh_node_idxs )

    
    xcollocation = mean.(mesh)

    @mytime "Nearby facets" begin

            # A hard minimum distance to force correction of nodes that "go to themselves"
        # To-do: note that one could pre-compute a stricter minimum than this one
        r_margin = 2edgelenstats(mesh).maximum + 2step(grid)
        r_max = include_nearest ? max(r_margin, r_near) : r_near

        tree = KDTree(xcollocation)

        areintersecting(nei1, nei2) = all(>(0), length.(intersect.(nei1, nei2)))
        
        nearby_facets :: Vector{Vector{UInt32}} = let
            # Maybe use Transducers.jl ?
            @run_and_gc ThreadsX.map(eachindex(mesh)) do i                 
                js = inrange(tree, xcollocation[i], r_max) .|> UInt32
                filter!(js) do j
                    norm(xcollocation[j] - xcollocation[i]) < r_near || areintersecting(neigh_node_idxs[i], neigh_node_idxs[j])
                end
                js
            end
        end

    end

    return nearby_facets
end

# Fixer

make_fixer(OP, mesh, PROJ; k, 
            quad=quad_default,
            verbose=false,
            nfargs...) = mytime("Fixer"; verbose) do

    
    grid = PROJ.interp.grid
    neighsidx = PROJ.interp.neighsidx

    # Get the nearby facets
    nearby_facets = get_nearby_facets(mesh, grid; neigh_node_idxs=neighsidx, verbose, nfargs...)

 
    # The first index i is the field point, the second j is the source.
    ijs = ((i, j) for i in eachindex(mesh) for j in nearby_facets[i])


    h_interp = PROJ.interp.hvecs
    h_proj   = PROJ.hvecs



    fun = potata_idx(OP, grid; k#=, dims=#)

    @mytime "Calculating the terms" begin

        vals = ThreadsX.map(ijs) do (i, j)
            
            nsrc = normal(mesh[j]) # This is sometimes not needed but I doubt it dominates the compute time
            nfld = normal(mesh[i]) # Same

            val :: ComplexF32 = @inbounds -sum(
                hi * fun(iy, ix, nsrc, nfld) * hp

                for ((hi, ix), (hp, iy)) in Iterators.product(
                    zip(h_interp[i], Iterators.product(neighsidx[i]...)), 
                    zip(h_proj[j], Iterators.product(neighsidx[j]...))) 
            )
                    
            val += operator_elem(OP, mesh[i], mesh[j]; k, quad)

            val
            
        end 
    end

    @mytime false "Creating sparse operator" s_fix = sparse(
        first.(ijs), 
        last.(ijs), 
        vals, 
        length(mesh), length(mesh)
    ) :: SparseMatrixCSC{ComplexF32, Int}

    return s_fix
end

"""
Return a function that, given 2 indices grid nodes and the normals
of the facets, returns the contribution of such nodes in the potential
of the operator.
"""
function potata_idx(::Type{L}, grid; k)

    cache = make_cache(L, grid; k)[1]

    # take the indices in the grid, return the contribution of iy on ix
    return (iy, ix, ny, nx) -> @inbounds(cache[(abs.(ix .- iy) .+ 1)...])
end

function potata_idx(::Type{M}, grid; k, dims=size(grid))
    
    caches = make_cache(M, grid; k, dims)

    # take the indices in the grid, return the contribution of iy on ix
    return function (iy, ix, ny, nx)
        Δidx = ix .- iy
        idx_cache = abs.(Δidx) .+ 1
        signs = sign.(Δidx) 

        chs = map(c -> c[idx_cache...], caches)

        (SMatrix{1, 3}(signs .* ny) * chs)[1]
    end

end

function potata_idx(::Type{Mt}, grid; k, dims=size(grid))

    caches = make_cache(Mt, grid; k, dims)

    # take the indices in the grid, return the contribution of iy on ix
    return function (iy, ix, ny, nx)
        Δidx = ix .- iy
        idx_cache = abs.(Δidx) .+ 1
        signs = sign.(Δidx) 

        chs = map(c -> c[idx_cache...], caches)

        (chs * SMatrix{3, 1}(signs .* nx))[1]
    end

end


function potata_idx(::Type{N}, grid; k, dims=size(grid))

    caches = make_cache(N, grid; k, dims)

    # take the indices in the grid, return the contribution of iy on ix
    return function (iy, ix, ny, nx)
        Δidx = ix .- iy
        idx_cache = abs.(Δidx) .+ 1

        signs = ifelse.(Δidx .< 0, 1, -1)

        chs = map(c -> c[idx_cache...], caches)

        (SMatrix{1, 3}(signs .* ny) * chs * SMatrix{3, 1}(signs .* nx))[1]

    end

end


"""
Array with values taken from a function on its indices which are evaluated lazily.
It's hard-coded to be <:AbstractArray{ComplexF32, 3}
"""
struct ThunkArray{F} <: AbstractArray{ComplexF32, 3}
    cache ::Array{Union{Missing, ComplexF32}, 3}
    fun    :: F
end
ThunkArray(fun, dims) = ThunkArray(
    Array{Union{Missing, ComplexF32}}(missing, dims),
    fun
    )
Base.size(ta::ThunkArray) = size(ta.cache)
function Base.getindex(ta::ThunkArray, idx...) :: ComplexF32
    buf = ta.cache[idx...]
    !ismissing(buf) && return buf

    out = ta.fun(idx)
    ta.cache[idx...] = out
    return out
end


"""
Make a SMatrix{ThunkArray} that lazily return the potentials of OP sampled
at multiples of `step(grid)`.
"""
make_cache(OP, grid; k, dims=size(grid)) = map(BEM.potential(OP; k)) do pot
    fun(idx) :: ComplexF32 = pot(step(grid) .* (SVector(idx) .- 1))
    ThunkArray(fun, dims)
end



# Old
#=
make_fixer(OP, mesh, PROJ; k, 
            quad=quad_default,
            verbose=false,
            nfargs...) = mytime("Fixer"; verbose) do


    # Get the nearby facets
    nearby_facets = get_nearby_facets(mesh, PROJ.interp.grid; neigh_node_idxs=PROJ.interp.neighsidx, verbose, nfargs...)

 
    # The first index i is the field point, the second j is the source.
    ijs = ((i, j) for i in eachindex(mesh) for j in nearby_facets[i])


    h_interp = PROJ.interp.hvecs
    h_proj   = PROJ.hvecs

    neighvals = get_neighbouring_nodes(PROJ.interp)

    fun = potcomponent(potential(OP; k)) # Maybe type unstable and bothers? 

    
    @mytime "Calculating the terms" vals = ThreadsX.map(ijs) do (i, j)
        
        nsrc = normal(mesh[j]) # This is sometimes not needed but I doubt it dominates the compute time
        nfld = normal(mesh[i]) # Same

        # To-do: Good speedups could be achieved if the following could be optimized further
        
        val :: ComplexF32 = @inbounds -sum(
            hi * fun(x - y, nsrc, nfld) * hp

            for ((hi, x), (hp, y)) in Iterators.product(zip(h_interp[i], neighvals[i]), zip(h_proj[j], neighvals[j])) 
        )
                
        val += operator_elem(OP, mesh[i], mesh[j]; k, quad)

        val
        
    end 

    @mytime false "Creating sparse operator" s_fix = sparse(
        first.(ijs), 
        last.(ijs), 
        vals, 
        length(mesh), length(mesh)
    ) :: SparseMatrixCSC{ComplexF32, Int}

    return s_fix
end

potcomponent(pots::SMatrix{1, 1}) = (Δx, nsrc, nfld) -> pots[1](Δx)
#=potcomponent(pots::SMatrix{3, 1}) = (Δx, nsrc, nfld) -> sum(pot(Δx) * ns for (ns, pot) in zip(nsrc, pots))
potcomponent(pots::SMatrix{1, 3}) = (Δx, nsrc, nfld) -> sum(pot(Δx) * nf for (nf, pot) in zip(nfld, pots))
potcomponent(pots::SMatrix{3, 3}) = (Δx, nsrc, nfld) -> sum(pot(Δx) * ns * nf for ((ns, nf), pot) in zip(Iterators.product(nsrc, nfld), pots))
=#
potcomponent(pots::SMatrix{3, 1}) = function (Δx, nsrc, nfld)
    ps = map(pot -> pot(Δx), pots)

    sum(pot(Δx) * ns for (ns, pot) in zip(nsrc, pots))
    first( SMatrix{1, 3}(nsrc) * ps )
end
potcomponent(pots::SMatrix{1, 3}) = function (Δx, nsrc, nfld)
    ps = map(pot -> pot(Δx), pots)

    first( ps * SMatrix{3, 1}(nfld))
end
potcomponent(pots::SMatrix{3, 3}) = function (Δx, nsrc, nfld)
    ps = map(pot -> pot(Δx), pots)

    first( SMatrix{1, 3}(nsrc) * ps * SMatrix{3, 1}(nfld))
end
=#