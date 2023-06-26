potential(::Type{L}; k, rmin=1e-5) = @SMatrix [ function (x)
    r = norm(x)
    out = green(r; k)
    return r < rmin ? zero(out) : out
end for _ in 1:1, _ in 1:1]


potential(::Type{M}; k, rmin=1e-5) = @SMatrix [

    function (x)
            r = norm(x)
            out = x[i] * green(r; k) * (1 - im * k * r) / r^2
            return r < rmin ? zero(out) : out
        end
        for i in 1:3, _ in 1:1
            
    ]


potential(::Type{Mt}; k, rmin=1e-5) = (-) .∘ permutedims(potential(M; k, rmin))




potential(::Type{N}; k, rmin=1e-5) = @SMatrix [
    function(xmy)
        r = norm(xmy)
        kr = k * r
        ikr = im * kr
        val = -(3 + (ikr - 3) * ikr) * xmy[j] * xmy[i] / r^4 + (i == j) * (1 - ikr) / r^2
        
        out = green(r; k) * val
        return r < rmin ? zero(out) : out
    end
    for i in 1:3, j in 1:3
]


#=
potential_v2(::Type{M}; k, rmin=1e-5) = @SMatrix [
        function (x)
            r = norm(x)
            out = der(Δyi -> green(Δyi * ydir, x; k), 0)
            return r < rmin ? zero(out) : out
        end
        for ydir in (@SVector([1, 0, 0]), @SVector([0, 1, 0]), @SVector([0, 0, 1])), _ in 1:1            
    ]

potential_v2(::Type{Mt}; k, rmin=1e-5) = [
    function (x)
        r = norm(x)
        out = der(Δxi -> green(@SVector([0, 0, 0]), x + Δxi * xdir; k), 0)
        return r < rmin ? zero(out) : out
    end
    for xdir in (@SVector([1, 0, 0]), @SVector([0, 1, 0]), @SVector([0, 0, 1]))
]

function potential_v2(::Type{N}; k, rmin=1e-5)
    function fun(i, x)
        r = norm(x)
        out = x[i] * green(r; k) * (1 - im * k * r) / r^2
        return r < rmin ? zero(out) : out
    end

    @SMatrix [

        x -> der(Δx -> fun(i, x + Δx * xdir), 0)
        for i in 1:3,
            xdir in (@SVector([1, 0, 0]), @SVector([0, 1, 0]), @SVector([0, 0, 1]))
    ]
end 

function potential_v3(::Type{N}; k, rmin=1e-5)
    function fun(i, x)
        r = norm(x)
        out = x[i] * green(r; k) * (1 - im * k * r) / r^2
        return r < rmin ? zero(out) : out
    end

    @SMatrix [
        
        function (x)
            r = norm(x)
            out = der(Δyi ->
                    der(Δxi -> green(Δyi * ydir, x + Δxi * xdir; k), 
                    0),
                    0)
            return r < rmin ? zero(out) : out
        end
        for ydir in (@SVector([1, 0, 0]), @SVector([0, 1, 0]), @SVector([0, 0, 1])),
            xdir in (@SVector([1, 0, 0]), @SVector([0, 1, 0]), @SVector([0, 0, 1]))
    ]
end 
=#

## Newest

"""
create_operator_approx(op<:Operator, mesh; k, λs_near=0.01, gridfactor=1, quad=quad_default, h_nterms=17, parallel_tasks=false, correction=true)

Return a LinearMap that that applies `op` to a vector of mesh values.
"""
create_operator_approx(OP, mesh; k, 
            λs_near=nothing, 
            r_near=nothing, 
            gridfactor=1, 
            quad=quad_default, 
            h_nterms=8, 
            include_nearest=true,
            verbose=false
            ) = mytime("Creating approximate operator"; verbose, gc=true) do

    edgelength = edgelenstats(mesh).deciles[9]
    grid = Grid(mesh; gridfactor, edgelength)
    verbose && @myinfo "Grid size: $(size(grid))"

    pots = potential(OP; k)

    @mytime "$OP" begin
        INTERP = Interpolator(mesh, grid; n=h_nterms, verbose)
        PROJ   = (size(pots, 1) == 1 ? ScalarProjector : VectorProjector)(mesh, INTERP; quad, verbose)
        CONVs  = @mytime "Convolvers" map(fun -> Convolver(fun, grid), pots) 
        FIX    = make_fixer(OP, mesh, PROJ; k, λs_near, r_near, include_nearest, verbose)
    end

    function s(σs)
        @assert length(σs) == length(mesh) "It's a mesh with $(mesh) facets, and you just input $(σs) weights"

        @mytime false "Applying" begin
            verbose && print('.')            
            σ_grids = PROJ * σs

            φ_grids = [ CONVs[i, j] * σ_grids[i] for i in 1:size(CONVs, 1), j in 1:size(CONVs, 2)] |> SMatrix{size(CONVs)...}

# Attempt at parallelizing the multiple convolutions            
#=            φ_grids = ThreadsX.map(CartesianIndices(CONVs)) do ci
                i, j = Tuple(ci)
                CONVs[i, j] * σ_grids[i]
            end |> SMatrix{size(CONVs)...}
=#
            φs_app = INTERP * φ_grids

            φs     = φs_app .+ FIX * σs
        end
        return φs
    end

    return LinearMap{ComplexF32}(s, length(mesh), length(mesh))
end


##
# Grid
struct Grid <: AbstractArray{Float32, 3}
    x :: typeof(2:2f0:10)
    y :: typeof(2:2f0:10)
    z :: typeof(2:2f0:10)
end

"""
    Grid(mesh; gridfactor=2, edgelength=[90% quantile])

Return a grid where 
"""
function Grid(mesh::AbstractMesh; gridfactor=0.5, edgelength=edgelenstats(mesh).deciles[9])
    Δr :: Float32 = edgelength / gridfactor

    # The grid; e.g., (0.2:Δr:0.4, 12.3:Δr:13:3, -2.2:Δr:-1.1)
    x, y, z = map(eachcoordinate(vertices(mesh))) do coords
        p0, pe = extrema(coords)
        range(p0 - Δr / 2; 
            stop = pe + Δr, 
            step = Δr   
            )
    end

    return Grid(x, y, z)
end

"""
    Grid(Δg, (Nx, Ny, Nz))
    Grid(Δg, (Nx, Ny, Nz), (x0, y0, z0))
"""
function Grid(Δg, (Nx, Ny, Nz)::NTuple{3, Integer}, (x0, y0, z0) = (0, 0, 0))
    Grid(
        range(x0; step=Δg, length=Nx),
        range(y0; step=Δg, length=Ny),
        range(z0; step=Δg, length=Nz)
    )
end

Base.getindex(g::Grid, i::Integer, j::Integer, k::Integer) = SVector(g.x[i], g.y[j], g.z[k])
Base.getindex(g::Grid, is, js, ks) = [ g[i, j, k] for i in is, j in js, k in ks ]
Base.step(g::Grid) = g.x.step
Base.size(g::Grid) = length.((g.x, g.y, g.z))

for fn in ("interpolator.jl", "projector.jl", "convolver.jl", "fixer.jl")
    include(joinpath("Accelerated", fn))
end

function computeneighbourhood(face::Face, grid)
    ns = map(v -> computeneighbourhood(v, grid), face)

    return Tuple(UnitRange(extrema(Iterators.flatten(coords))...) for coords in eachcoordinate(ns))
end
computeneighbourhood(vert::Vertex, grid::Grid) = map(computeneighbourhood, vert, (grid.x, grid.y, grid.z)) 
computeneighbourhood(val::Number, ran) = searchsortedfirst(ran, val) .+ (-1, 0)

# Depende de otros métodos que actualmente están en la parte older


##================================================================
##================================================================
## Older version =================================================

#=

function create_operator_approx_old(::Type{L}, mesh; k, λs_near=0.01, gridfactor=2, quad=quad_default, h_nterms=8, verbose=false)

    grid = Grid(mesh; gridfactor)

    pot = potential(L; k)

    @timeit to "Pre-processing" begin
        @timeit to "Interpolator"  INTERP = Interpolator(mesh, grid; n=h_nterms)
        @timeit to "Projector"     PROJ   = ScalarProjector(mesh, INTERP; quad)
        @timeit to "Convolver"     CONV   = Convolver(pot, grid)
        @timeit to "Fixer"         FIX    = make_fixer(mesh, pot, PROJ; r_near=λs_near * 2π / k)
    end

    function s(σs)
        @assert length(σs) == length(mesh) "It's a mesh with $(mesh) facets, and you just input $(σs) weights"

        verbose && @info "Projecting..."
        @timeit to "Projecting"            q_grid = PROJ * σs

        verbose && @info "Convolving..."
        @timeit to "Convolving"            φ_grid = CONV * q_grid

        verbose && @info "Interpolating..."
        @timeit to "Interpolating"         φs_app = INTERP * φ_grid

        verbose && @info "Fixing..."
        @timeit to "Fixing"                φs     = φs_app .+ FIX * σs

        return φs
    end

    return LinearMap{ComplexF32}(s, length(mesh), length(mesh))
end


function create_operator_approx_old(::Type{M}, mesh; k, λs_near=0.01, gridfactor=2, quad=quad_default, h_nterms=8, verbose=false)


    grid = Grid(mesh; gridfactor)

    pots = potential(M; k)

    @timeit to "Pre-processing" begin
        @timeit to "Interpolator"  INTERP = Interpolator(mesh, grid; n=h_nterms)
        @timeit to "Projector"     PROJ   = VectorProjector(mesh, INTERP; quad)
        @timeit to "Convolver"     CONVs   = [ Convolver(pot, grid) for pot in pots ]
        # @timeit to "Fixer"         FIX    = make_fixer(mesh, pot, PROJ; r_near=λs_near * 2π / k)
    end

    function s(σs)
        @assert length(σs) == length(mesh) "It's a mesh with $(mesh) facets, and you just input $(σs) weights"

        verbose && @info "Projecting..."
        @timeit to "Projecting"            q_grids = PROJ * σs

        verbose && @info "Convolving..."
        @timeit to "Convolving"            φ_grid = sum( CONV * q_grid for (CONV, q_grid) in zip(CONVs, q_grids))

        verbose && @info "Interpolating..."
        @timeit to "Interpolating"         φs_app = INTERP * φ_grid

        verbose && @info "Fixing..."
        @timeit to "Fixing"                φs     = φs_app #.+ FIX * σs

        return φs
    end

    return LinearMap{ComplexF32}(s, length(mesh), length(mesh))
end

function create_operator_approx_old(::Type{Mt}, mesh; k, λs_near=0.01, gridfactor=2, quad=quad_default, h_nterms=8, verbose=false)

    grid = Grid(mesh; gridfactor)

    pots = potential(Mt; k)

    @timeit to "Pre-processing" begin
        @timeit to "Interpolator"  INTERP = Interpolator(mesh, grid; n=h_nterms)
        @timeit to "Projector"     PROJ   = ScalarProjector(mesh, INTERP; quad)
        @timeit to "Convolver"     CONVs   = [ Convolver(pot, grid) for pot in pots ]
        # @timeit to "Fixer"         FIX    = make_fixer(mesh, pot, PROJ; r_near=λs_near * 2π / k)
    end

    function s(σs)
        @assert length(σs) == length(mesh) "It's a mesh with $(mesh) facets, and you just input $(σs) weights"

        verbose && @info "Projecting..."
        @timeit to "Projecting"            q_grid = PROJ * σs

        verbose && @info "Convolving..."
        @timeit to "Convolving"            φ_grids = [ CONV * q_grid for CONV in CONVs ]

        verbose && @info "Interpolating..."
        @timeit to "Interpolating"         φs_app = INTERP * φ_grids

        verbose && @info "Fixing..."
        # @timeit to "Fixing"                φs     = φs_app .+ FIX * σs

        return φs
    end

    return LinearMap{ComplexF32}(s, length(mesh), length(mesh))
end

function create_operator_approx_old(::Type{N}, mesh; k, λs_near=0.01, gridfactor=2, quad=quad_default, h_nterms=8, verbose=false)

    grid = Grid(mesh; gridfactor)

    pots = potential(N; k)

    @timeit to "Pre-processing" begin
        @timeit to "Interpolator"  INTERP  = Interpolator(mesh, grid; n=h_nterms)
        @timeit to "Projector"     PROJ    = VectorProjector(mesh, INTERP; quad)
        @timeit to "Convolver"     CONVs   = [ Convolver(pot, grid) for pot in pots ]
        # @timeit to "Fixer"         FIX    = make_fixer(mesh, pot, PROJ; r_near=λs_near * 2π / k)
    end

    function s(σs)
        @assert length(σs) == length(mesh) "It's a mesh with $(mesh) facets, and you just input $(σs) weights"

        verbose && @info "Projecting..."
        @timeit to "Projecting"            q_grids = PROJ * σs

        verbose && @info "Convolving..."
        @timeit to "Convolving"            φ_grids = [ CONVs[i, j] * q_grids[i] for i in 1:3, j in 1:3 ]

        verbose && @info "Interpolating..."
        @timeit to "Interpolating"         φs_app = sum(INTERP * φ_grid_row for φ_grid_row in eachrow(φ_grids) )

        verbose && @info "Fixing..."
        # @timeit to "Fixing"                φs     = φs_app .+ FIX * σs

        return φs
    end

    return LinearMap{ComplexF32}(s, length(mesh), length(mesh))
end






"""
    create_operator_approx_original(op<:Operator, mesh; k, λs_near=0.01, gridfactor=1, quad=quad_default, h_nterms=17, parallel_tasks=false, correction=true)

    Return a LinearMap that that applies `op` to a vector of mesh values.
"""
function create_operator_approx_original(::Type{L}, mesh; k, λs_near=0.01, gridfactor=2, quad=quad_default, h_nterms=8, parallel_tasks=false, correction=true)
    r_near = λs_near * 2π / k
    @show r_near
    # The grid step, and an upper bound to the minimum radius of the circumcircles
    local Δr, rmin
    let
        edgelens = [norm(v2 .- v1) for face in mesh for (v1, v2) in subsets(face, 2)]
        
        Δr =  quantile(edgelens, 0.99) / gridfactor
        rmin = if correction
                1e-10
                #=max(
                    minimum(edgelens) / 2sqrt(3),
                    r_near - maximum(edgelens) - 2Δr - eps(Δr)
                ) / 10  # solo por las recontra dudas, meh
                =#
            # rmin = 1e-3
        else
            rmin = r_near
        end
    end
    

    # The centroids of each facet
    xcollocation = mean.(mesh)

    # The grid; e.g., (0.2:Δr:0.4, 12.3:Δr:13:3, -2.2:Δr:-1.1)
    grid = map(eachcoordinate(vertices(mesh))) do coords
        p0, pe = extrema(coords)
        range(p0 - Δr / 2; 
            stop = pe + Δr, 
            step = Δr   
            )
    end
    @show extrema.(grid)
    @show length.(grid)

    # The neighbourhoods, given as vectors of indices
    local neighsidx::Vector{NTuple{3, UnitRange{Int64}}}, neighvals::Vector{Vector{SVector{3, Float64}}}, nearby_facets
    task_A = @task begin
        neighsidx = ThreadsX.map(f -> computeneighbourhood(f, grid), mesh)
        neighvals = ThreadsX.map(nei -> getneighval(nei, grid), neighsidx)

    end
    task_B = @task begin 
        # nearby_facets = [inrange(KDTree(xcollocation), pt, r_near) for pt in xcollocation]
        nearby_facets = convert.(Vector{UInt32}, 
            inrange(KDTree(xcollocation), xcollocation, r_near) 
        )
    end
  
        # Interpolation functions
    local hs::Vector{Vector{Float64}}
    hints = Vector{Float64}[ sizehint!(Float64[], 27) for _ in mesh ]
    local hfs
    task_CD = @task begin
        hfs = h.(neighvals; n=h_nterms)
    end
        # Interpolation operator
    task_C(hfs) = @task begin
        hs = [ hf(ξ) for (hf, ξ) in zip(hfs, xcollocation) ]
    end
        # Projection operator
    task_D(hfs) = @task begin
        #hints = [ integrate_triangle(y -> hf(y), facet; quad) for (hf, facet) in zip(hfs, mesh)]
        hints = ThreadsX.collect( integrate_triangle(hf, facet; quad) for (hf, facet) in zip(hfs, mesh) )
        #=@inbounds Threads.@threads for i in eachindex(mesh)
            #
        end=#
    end
    
        
    function project(σs)
        q_grid = zeros(ComplexF64, length.(grid)...)

        ThreadsX.foreach(σs, hints, neighsidx) do σ, hint, ineigh
            q_grid[CartesianIndices(ineigh) |> vec] .+= σ .* hint
        end

        return q_grid
    end

    function interp(φ_grid)
        φs = [ dot(h, φ_grid[CartesianIndices(ineigh)]) for (h, ineigh) in zip(hs, neighsidx)]

        return φs
    end
        
    function greenify end
    task_E = @task begin
        q_grid_size = length.(grid)
        npad = 2 .* q_grid_size .- 1
        # Sí, esto quedó asqueroso. Cuando la evaluás en un índice mayor al tamaño de `q`, lo redirecciona como para que sea circular
        # Difícil de explicar rapidito en un comentario
        function gidx(ci::CartesianIndex)
            idx = map(fixind, Tuple(ci), npad)
            r = Δr * norm(idx)
            out = green_trunc(r; k=k, rmin) 
            return out
        end
        function fixind(i, s) ::Int
            from  = -(s - 1) ÷ 2
            to    = from + s - 1
            return mod(i - 1, from : to)
        end
        g_arr = [ gidx(ind) for ind in CartesianIndices(npad) ]
        gf    = fft(g_arr); 
        FFT! = plan_fft!(Array{ComplexF64}(undef, npad))
        IFFT! = plan_ifft!(Array{ComplexF64}(undef, npad)) #inv(FFT!)
            # Preallocations
        gf_prealloc = similar(gf); q_grid_pad = zeros(ComplexF64, npad)
        function greenify(q_grid)
            @assert size(q_grid) == q_grid_size

            begin
                q_grid_pad .= 0
                q_grid_pad[CartesianIndices(q_grid)] .= q_grid
            end

            out   = let
                @timeit to "FTT" FFT! * q_grid_pad
                @timeit to "Product" gf_prealloc .=  q_grid_pad .* gf
                @timeit to "IFFT" IFFT! * gf_prealloc
            end

            return out[CartesianIndices(q_grid)] # lo trunco al tamaño de q
        end
    end


    # Precompute the sparse operator just for the wrong ones.
        # The first index i is the field point, the second j is the source.
    local s_fix
    task_F = @task begin
        ijs = ((i, j) for i in eachindex(mesh) for j in nearby_facets[i])


        @timeit to "Calculating the terms" vals = ThreadsX.map(ijs) do (i, j)
            local val :: ComplexF64

            if correction
                # @inbounds gmat = collect( green(x, y; k, rmin) for x in neighvals[i], y in neighvals[j])
                # @inbounds val = - dot(hs[i], gmat, hints[j])
                val = @inbounds -sum( h * (green_trunc(x, y; k)#=::ComplexF64=#) * hint
                    #=for (h, x) in zip(hs[i], neighvals[i])
                    for (hint, y) in zip(hints[j], neighvals[j])=#
                    for ((h, x), (hint, y)) in Iterators.product(zip(hs[i], neighvals[i]), zip(hints[j], neighvals[j]))
                    )
                    
            else
                val = zero(ComplexF64)
            end :: ComplexF64

                # I think maybe this exact part isn't parallelizing well
            @inbounds val += i == j ? (intreg(k, mesh[j])::ComplexF64) : (integrate_triangle(y -> green_trunc(xcollocation[i], y; k), mesh[j]; quad)::ComplexF64)

            val :: ComplexF64
        end
        @timeit to "Creating sparse operator" s_fix = sparse(
            first.(ijs), 
            last.(ijs), 
            vals, 
            length(mesh), length(mesh)
        ) :: SparseMatrixCSC{ComplexF64, Int}
    end

    #=
    A: neighbouring nodes
    B: neighrbouring facets
    C: interpolation operator
    D: projection operator
    E: convolution
    F: correction 

    A -> CD
    CD -> C
    CD -> D
    (B, C, D) -> F
    E
    =#
    if parallel_tasks == false 
        println("Computing neighbouring nodes...")
        @time @timeit to "Neighbouring nodes" task_A |> schedule |> wait

        println("Computing interpolation functions...")        
        @time @timeit to "Interpolation functions" task_CD |> schedule |> wait
            println("There are $(round(mean(prod(length.(nei)) for nei in neighsidx); digits=2)) neighbouring grid nodes per facet on average.")
            
        println("Computing neighbouring facets...")
        @time @timeit to "Neighbouring facets" task_B |> schedule |> wait
            println("$(round(sum(length.(nearby_facets)) / length(mesh)/length(mesh)*100;digits=4))% of facet pairs will be fixed.")
        println("Computing interpolation operator...")
        @time @timeit to "Interpolation op" task_C(hfs) |> schedule |> wait
        println("Computing projection operator...")
        @time @timeit to "Projection op" task_D(hfs) |> schedule |> wait
        #=println("Computing interp + proj op")
        @time @timeit to "Projection + interp op and functions" task_CDCD |> schedule |> wait=#
        println("Computing convolution operator...")
        @time @timeit to "Convolution" task_E |> schedule |> wait
        println("Computing correction operator...")
        @time @timeit to "Correction" task_F |> schedule |> wait
    else 
        println("Starting Nnodes, Nfacets and conv")
        schedule.([task_A, task_B, task_E])
        println("Done scheduling them")
        @time wait(task_A); println("Done with Nnodes")
        println("Starting interFun")
        schedule(task_CD)
        @time wait(task_CD); println("Done with interFun")
        schedule.([task_C, task_D]); println("Starting interOp and projOp")
        @time wait.([task_B, task_C, task_D]); println("Done with NFacets, interOp ,and projOp")
        schedule(task_F); println("Starting correction")
        @time wait.([task_E, task_F]); println("Done with conv and correction")
    end


    function s(σs)
        @assert length(σs) == length(mesh) "It's a mesh with $(mesh) facets, and you just input $(σs) weights"

        @timeit to "Projection"            q_grid = project(σs)
        @timeit to "Convolution"           φ_grid = greenify(q_grid)
        @timeit to "Interpolation"         φs_app = interp(φ_grid)
        @timeit to "Near-field correction" φs     = φs_app .+ s_fix * σs

        return φs
    end

    return LinearMap{ComplexF64}(s, length(mesh), length(mesh))
end

function create_operator_approx_original(::Type{M}, mesh; k, λs_near=0.01, gridfactor=1, quad=quad_default, h_nterms=17, parallel_tasks=false, correction=true)
    r_near = λs_near * 2π / k
    @show r_near
    # The grid step, and an upper bound to the minimum radius of the circumcircles
    local Δr, rmin
    let
        edgelens = [norm(v2 .- v1) for face in mesh for (v1, v2) in subsets(face, 2)]
        
        Δr =  quantile(edgelens, 0.99) / gridfactor
        rmin = if correction
                1e-10
                #=max(
                    minimum(edgelens) / 2sqrt(3),
                    r_near - maximum(edgelens) - 2Δr - eps(Δr)
                ) / 10  # solo por las recontra dudas, meh
                =#
            # rmin = 1e-3
        else
            rmin = r_near
        end
    end
    

    # The centroids of each facet
    xcollocation = mean.(mesh)

    # The grid; e.g., (0.2:Δr:0.4, 12.3:Δr:13:3, -2.2:Δr:-1.1)
    grid = map(eachcoordinate(vertices(mesh))) do coords
        p0, pe = extrema(coords)
        range(p0 - Δr / 2; 
            stop = pe + Δr, 
            step = Δr   
            )
    end
    @show extrema.(grid)
    @show length.(grid)

    # The neighbourhoods, given as vectors of indices
    # local neighsidx::Vector{NTuple{3, UnitRange{Int64}}}, neighvals::Vector{Vector{SVector{3, Float64}}}, nearby_facets
    global neighsidx, neighvals, nearby_facets    
    task_A = @task begin
        neighsidx = ThreadsX.map(facet -> computeneighbourhood(facet, grid), mesh)
        neighvals = ThreadsX.map(nei -> getneighval(nei, grid), neighsidx)

    end
    task_B = @task begin 
        # nearby_facets = [inrange(KDTree(xcollocation), pt, r_near) for pt in xcollocation]
        nearby_facets = convert.(Vector{UInt32}, 
            inrange(KDTree(xcollocation), xcollocation, r_near) 
        )
    end

        # Interpolation functions
    local hs::Vector{Vector{Float64}}
    local hfs
    task_CD = @task begin
        hfs = h.(neighvals; n=h_nterms)
    end
        # Interpolation operator
    task_C(hfs) = @task begin
        hs = [ hf(ξ) for (hf, ξ) in zip(hfs, xcollocation) ]
    end
    

        # Projection weights
    hints = Vector{Float64}[ sizehint!(Float64[], 27) for _ in mesh ]
    task_D(hfs) = @task begin
        #hints = [ integrate_triangle(y -> hf(y), facet; quad) for (hf, facet) in zip(hfs, mesh)]
        hints = ThreadsX.collect( integrate_triangle(hf, facet; quad) for (hf, facet) in zip(hfs, mesh) )
        #=@inbounds Threads.@threads for i in eachindex(mesh)
            #
        end=#
    end
    
        
    function project(σs)
        q_grids = [ zeros(ComplexF64, length.(grid)...) for _ in 1:3 ]

        # ThreadsX.foreach(σs, hints, neighsidx, mesh) do σ, hint, ineigh, facet
        for (σ, hint, ineigh, facet) in zip(σs, hints, neighsidx, mesh)
            n = normal(facet)
            for i in 1:3
                q_grids[i][CartesianIndices(ineigh) |> vec] .+= n[i] * σ .* hint
            end
        end

        return q_grids
    end

    function interp(φ_grid)
        φs = [ dot(h, φ_grid[CartesianIndices(ineigh)]) for (h, ineigh) in zip(hs, neighsidx)]

        return φs
    end
        
    function greenify end
    task_E = @task begin
        q_grid_size = length.(grid)
        npad = 2 .* q_grid_size .- 1
        # Sí, esto quedó asqueroso. Cuando la evaluás en un índice mayor al tamaño de `q`, lo redirecciona como para que sea circular
        # Difícil de explicar rapidito en un comentario
        function gidx(ci::CartesianIndex)
            x = Δr .* map(fixind, SVector(Tuple(ci)), npad)
            # r = Δr * norm(idx)

            out = grad(x -> green_trunc(x, @SVector([0, 0, 0]); k=k, rmin), x) 
            return out
        end
        function fixind(i, s) ::Int
            from  = -(s - 1) ÷ 2
            to    = from + s - 1
            return mod(i - 1, from : to)
        end
        g_arrs = [ gidx(ind) for ind in CartesianIndices(npad) ]
        gfs    = [ fft(getindex.(g_arrs, i)) for i in 1:3]

        FFT!  = plan_fft!(Array{ComplexF64}(undef, npad))
        IFFT! = plan_ifft!(Array{ComplexF64}(undef, npad)) #inv(FFT!)

            # Preallocations
        gf_prealloc = similar(first(gfs)); q_grid_pad = zeros(ComplexF64, npad)
        function greenify(q_grids)
            @assert all(q_grid -> size(q_grid) == q_grid_size, q_grids)

            sum(zip(q_grids, gfs)) do (q_grid, gf)

                    q_grid_pad .= 0
                    q_grid_pad[CartesianIndices(q_grid)] .= q_grid

                    out   = let
                        @timeit to "FTT" FFT! * q_grid_pad
                        @timeit to "Product" gf_prealloc .=  q_grid_pad .* gf
                        @timeit to "IFFT" IFFT! * gf_prealloc
                    end

                    # This allocates
                    out[CartesianIndices(q_grid)] # lo trunco al tamaño de q
            end
        end
    end



    # Precompute the sparse operator just for the wrong ones.
        # The first index i is the field point, the second j is the source.
    local s_fix
    task_F = @task begin
        ijs = ((i, j) for i in eachindex(mesh) for j in nearby_facets[i])


        @timeit to "Calculating the terms" vals = ThreadsX.map(ijs) do (i, j)
            local val :: ComplexF64

            if correction
                # @inbounds gmat = collect( green(x, y; k, rmin) for x in neighvals[i], y in neighvals[j])
                # @inbounds val = - dot(hs[i], gmat, hints[j])
                val = @inbounds -sum( h * ddir(yy -> green_trunc(x, yy; k), y, normal(mesh[j])) * hint # WASTE
                    #=for (h, x) in zip(hs[i], neighvals[i])
                    for (hint, y) in zip(hints[j], neighvals[j])=#
                    for ((h, x), (hint, y, n)) in Iterators.product(
                        zip(hs[i], neighvals[i]), 
                        zip(hints[j], neighvals[j]))
                    )
                    
            else
                val = zero(ComplexF64)
            end :: ComplexF64

                # I think maybe this exact part isn't parallelizing well
            i == j && return val
            
            # Integro en el mesh[j], en el punto xcollocation[i]
            n = normal(mesh[j])
            val + integrate_triangle(mesh[j]; quad) do y
                dot(
                    n, 
                    grad(yy -> green(xcollocation[i], yy; k), y))
            end

            
        end
        @timeit to "Creating sparse operator" s_fix = sparse(
            first.(ijs), 
            last.(ijs), 
            vals, 
            length(mesh), length(mesh)
        ) :: SparseMatrixCSC{ComplexF64, Int}
    end


    #=
    A: neighbouring nodes
    B: neighrbouring facets
    C: interpolation operator
    D: projection operator
    E: convolution
    F: correction 

    A -> CD
    CD -> C
    CD -> D
    (B, C, D) -> F
    E
    =#
    if parallel_tasks == false 
        println("Computing neighbouring nodes...")
        @time @timeit to "Neighbouring nodes" task_A |> schedule |> wait

        println("Computing interpolation functions...")        
        @time @timeit to "Interpolation functions" task_CD |> schedule |> wait
            println("There are $(round(mean(prod(length.(nei)) for nei in neighsidx); digits=2)) neighbouring grid nodes per facet on average.")
            
        println("Computing neighbouring facets...")
        @time @timeit to "Neighbouring facets" task_B |> schedule |> wait
            println("$(round(sum(length.(nearby_facets)) / length(mesh)/length(mesh)*100;digits=4))% of facet pairs will be fixed.")
        println("Computing interpolation operator...")
        @time @timeit to "Interpolation op" task_C(hfs) |> schedule |> wait
        println("Computing projection operator...")
        @time @timeit to "Projection op" task_D(hfs) |> schedule |> wait
        #=println("Computing interp + proj op")
        @time @timeit to "Projection + interp op and functions" task_CDCD |> schedule |> wait=#
        println("Computing convolution operator...")
        @time @timeit to "Convolution" task_E |> schedule |> wait
        println("Computing correction operator...")
        @time @timeit to "Correction" task_F |> schedule |> wait
    else 
        println("Starting Nnodes, Nfacets and conv")
        schedule.([task_A, task_B, task_E])
        println("Done scheduling them")
        @time wait(task_A); println("Done with Nnodes")
        println("Starting interFun")
        schedule(task_CD)
        @time wait(task_CD); println("Done with interFun")
        schedule.([task_C, task_D]); println("Starting interOp and projOp")
        @time wait.([task_B, task_C, task_D]); println("Done with NFacets, interOp ,and projOp")
        schedule(task_F); println("Starting correction")
        @time wait.([task_E, task_F]); println("Done with conv and correction")
    end


    function s(σs)
        @assert length(σs) == length(mesh) "It's a mesh with $(mesh) facets, and you just input $(σs) weights"

        @timeit to "Projection"            q_grid = project(σs)
        @timeit to "Convolution"           φ_grid = greenify(q_grid)
        @timeit to "Interpolation"         φs_app = interp(φ_grid)
        @timeit to "Near-field correction" φs     = φs_app .+ s_fix * σs

        return φs
    end

    return LinearMap{ComplexF64}(s, length(mesh), length(mesh))
end



computeneighbourhood(vert::Vertex, grid::Tuple) = map(computeneighbourhood, vert, grid) 

green_trunc(x, y; k, krmin=1e-10, rmin=1e-5) = green_trunc(norm(x .- y); k, krmin, rmin)
function green_trunc(r; k, krmin=1e-5, rmin=1e-5)
    out = green(r; k, krmin)
    return r < rmin ? zero(out) : out
end



getneighval(neigh, grid::Tuple) = SVector.(vec(collect(product(getindex.(grid, neigh)...))))

=#





