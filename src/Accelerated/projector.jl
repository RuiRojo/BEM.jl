
# Projector
abstract type Projector end
struct ScalarProjector <: Projector
    interp :: Interpolator
    hvecs  :: Vector{Vector{Float32}}
    σ_grid :: Array{ComplexF32, 3}# Just a pre-allocation
end

ScalarProjector(mesh::LightMesh; quad=quad_default, kwargs...) = ScalarProjector(mesh, Interpolator(mesh; kwargs...); quad)

push!(MEMO_FUNS, ScalarProjector)
@memoize ScalarProjector(mesh::LightMesh, interp::Interpolator; quad=quad_default, verbose=false) = mytime("Projector"; verbose) do
    hfs = interp.hfuns

    # hvecs = ThreadsX.collect(Vector{Float32}, integrate_triangle(hf, facet; quad) for (hf, facet) in zip(hfs, mesh) )
    hvecs = ThreadsX.map(hfs, mesh) do hf, facet
        integrate_triangle(hf, facet; quad) .|> Float32
    end .|> Vector
    
    σ_grid = zeros(ComplexF32, size(interp.grid))

    return ScalarProjector(interp, hvecs, σ_grid)
end

Base.:*(proj::ScalarProjector, σs) = @mytime "Projecting" begin
    grid = proj.interp.grid
    hvecs = proj.hvecs
    neighsidx = proj.interp.neighsidx

    σ_grid = proj.σ_grid
    σ_grid .= 0

        # Innocently parallelizing this causes race conditions
    #=ThreadsX.=#foreach(σs, hvecs, neighsidx) do σ, hvec, ineigh
        σ_grid[CartesianIndices(ineigh) |> vec] .+= σ .* hvec
    end

    return tuple(σ_grid)

end

struct VectorProjector <: Projector
    interp :: Interpolator
    hvecs  :: Vector{Vector{Float32}}
    σ_grids :: NTuple{3, Array{ComplexF32, 3}}# Just a pre-allocation
    normals :: Vector{SVector{3, Float32}}
end
VectorProjector(mesh::LightMesh; kwargs...) = VectorProjector(mesh, Interpolator(mesh; kwargs...))
function VectorProjector(mesh::LightMesh, interp::Interpolator; quad=quad_default, verbose=false) 

    interp, hvecs = let
        sp = ScalarProjector(mesh, interp; quad, verbose)
        sp.interp, sp.hvecs
    end

    σ_grids = Tuple(zeros(ComplexF32, size(interp.grid)) for _ in 1:3 )

    return VectorProjector(interp, hvecs, σ_grids, normal.(mesh))
end

Base.:*(proj::VectorProjector, σs) = @mytime "Projecting" begin
    grid = proj.interp.grid
    hvecs = proj.hvecs
    neighsidx = proj.interp.neighsidx

    σ_grids = proj.σ_grids
    for σ_grid in σ_grids
        σ_grid .= 0
    end

    # Innocent thread parallelization leads to race conditions and non-deterministic results
    #=ThreadsX.=#foreach(σs, hvecs, neighsidx, proj.normals) do σ, hvec, ineigh, normal
        v = σ .* hvec
        for (n, σ_grid) in zip(normal, σ_grids)
            σ_grid[CartesianIndices(ineigh) |> vec] .+= v * n
        end
    end

    return σ_grids
end