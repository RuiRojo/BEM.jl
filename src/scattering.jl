
"""
    intensity(c::Complex)

Return abs(c)^2.

It fails for real numbers in case they already represent intensities.
"""
intensity(c::Complex) = abs(c)^2

"""
    target_strength(c::Complex)
    target_strength(r::Real)

Return TS.

If the argument is real, it is interpreted as an intensity.
"""
target_strength(c::Complex) = target_strength(intensity(c))
target_strength(r::Real) = 10log10(r)




"""
    backscattering(bv, incidence_dir=Versor(0, 0), k, quad=quad_default, kwargs...)
    backscattering(mesh, incidence_dir=Versor(0, 0); kwargs...)
    backscattering(mesh, incidence_dirs_iter; kwargs...)

Return the backscattering in the farfield.

It can receive pre-computed `bv::BoundaryValues` or a `mesh` and a _kwarg_ `k`.
"""
backscattering(bv::BoundaryValues, idir::Versor; kwargs...) = bv(-idir; kwargs...)
function backscattering(mesh::AbstractMesh, idir::Versor; k, quad=quad_default, kwargs...)
    backscattering(
        scattering_boundary(mesh, idir; k, kwargs...),
        idir; 
        quad
    )
end
backscattering(mesh::AbstractMesh, idirs; 
    k, verbose=false, 
    quad=quad_default, 
    kwargs...) = run_and_gc() do
    
    bvs = scattering_boundary(mesh, idirs; k, kwargs...)

        # A progress logger should be added here, but ProgressLogging.jl doesn't work with `map`
    @mytime "Computing boundary and backscattering" @progress "Sweeping incidence dirs" out = [
        backscattering(take!(bvs), idir; quad)
        for idir in idirs
    ]
    return out
end



"""
    forward_scattering(bv::BoundaryValues; incidence_dir=Versor(0, 0), kwargs...)
    forward_scattering(mesh::AbstractMesh; k, incidence_dir=Versor(0, 0), quad=quad_default, kwargs...)

Return the forward scattering in the farfield.
It can receive pre-computed `bv::BoundaryValues` or a `mesh` and a _kwarg_ `k`.
"""
forward_scattering(bv::BoundaryValues, incidence_dir::Versor; kwargs...) = bv(incidence_dir; kwargs...)
function forward_scattering(mesh::AbstractMesh, incidence_dir::Versor; k, verbose=false, quad=quad_default, kwargs...)
    forward_scattering(
        scattering_boundary(mesh, incidence_dir; k, kwargs...), incidence_dir; quad, verbose
    )
end
forward_scattering(mesh::AbstractMesh, idirs; 
    k, verbose=false, 
    quad=quad_default, 
    kwargs...) = run_and_gc() do
    
    bvs = scattering_boundary(mesh, idirs; k, kwargs...)
#=
    out = @mytime "Computing boundary and forward scattering" map(idirs, bvs) do idir, bv
            forward_scattering(bv, idir; quad)
    end=#
    @mytime "Computing boundary and forward scattering" @progress "Sweeping incidence dirs" out = [
        forward_scattering(take!(bvs), idir; quad)
        for idir in idirs
    ]


    return out
end



# If a vector of directions is passed, `solver` needs to work on matrices
"""
    scattering_boundary(mesh, incidence_dir=Versor(0, 0); 
        k, 
        b_builder, 
        φ_interp,
        lm_builder = (mesh; k) -> create_operator_exact_matrix(L, mesh; k, quad), 
        solver= (\\),  
        quad=quad_default,
        )

Compute `BoundaryValues` for the given `mesh`, `k`, and `incidence_dir`.

The other other parameters are usually computed with `BEM.pars`.

Take the system Aφ = b, where A is the linear map `lm`.

- `b_builder`: a function of (p, v; k) that returns `b` given the incident 
    pressure `p` and external normal pressure derivative `v`.
- `φ_interp`: a function of (φ; k) that returns a vector of pairs (φL, φM), 
    translating vecto `φ` into monopolar densities φL and dipolar densities φM.
- `lm_builder`: a function of (mesh; k) that returns the linear map `A`.
- `solver`: the function that solves the linear system. Typical values include `(\\)` or `gmres`.
"""
scattering_boundary(mesh, incidence_dir::Versor; kwargs...) = with_memoize() do
    scattering_boundary(mesh; kwargs...)(incidence_dir)
end

#=function scattering_boundary(mesh, incidence_dir=Versor(0, 0); k, b_builder, φ_interp, 
    quad=quad_default,
    lm_builder = (mesh; k) -> create_operator_exact_matrix(L, mesh; k, quad), 
    solver= (\))	


    @timeit to "Computing boundary" begin

        @timeit to "Creating operator" begin
            # Linear map
            A = lm_builder(mesh; k)
        end

        @timeit to "Computing incident field in boundary" begin
            b = binc(mesh, k, incidence_dir, b_builder)
        end

        @timeit to "Solving system" begin
            # Solve A φ = binc
            φs = solver(A, b)
        end

        return if incidence_dir isa Versor
            BoundaryValues(;mesh, densities=φ_interp(φs; k), k)
        else
            map(eachcol(φs)) do φs
                BoundaryValues(;mesh, densities=φ_interp(φs; k), k)
            end
        end
    end
    
end
=#

"""
    scattering_boundary(mesh; kwargs...)

Like `scattering_boundary(mesh, incidence_dir; kwargs...)` but returns
a curried function that expects `incidence_dir` as argument.

# Keyword arguments

- formulation: This can be DirectFormulation() or IndirectFormulation(), which can receive
the μ parameter as argument. It can also be a SurfaceIntegralOperator like L/M/Mt/N. Note that
L is only possible for Dirichlet BC and N for Neumann. 
"""
function scattering_boundary(mesh; k, BC, 
    formulation=IndirectFormulation, μ=nothing, 
    algorithm=Exact(), solver=def_solver(algorithm), 
    verbose=false
    )

    ;b_builder, φ_interp, lm_grtor = pars(BC, formulation, μ) 
    lm_builder = lm_grtor(algorithm)

    return _scattering_boundary(mesh; k, 
        b_builder, 
        φ_interp, 
        verbose,
        lm_builder, 
        solver
    ) 
end

def_solver(::Type{Exact{Matrix}}) = (/)
def_solver(::Type{Exact{LinearMap}}) = gmres 
def_solver(::Type{Accelerated}) = gmres
def_solver(_) = gmres


function _scattering_boundary(mesh; k, b_builder, φ_interp, 
    verbose=false,
    lm_builder, 
    solver= (\)) 

    verbose && @myinfo "The maximum edge length is $(round(facet_vs_λ(mesh; k); digits=2)) times λ"
    
    with_memoize() do 
        @mytime false "Computing boundary" begin
            @mytime "Creating operator A" begin
                # Linear map
                A_raw = lm_builder(mesh; k)
                    
                    # tmp: special treatment to optimize for repeated matrix solutions (same mesh and k, different incidences)
                A = solver isa typeof(\) ? lu(A_raw) : A_raw
            end
        end

        return incidence_dir -> run_and_gc() do
            @mytime "Computing boundary" begin
                @mytime false "Computing incident field in boundary" begin
                    b = binc(mesh, k, incidence_dir, b_builder)
                end
                
                @mytime "Solving system" begin
                    # Solve A φ = binc
                    φs = solver(A, b)
                end
            end

            return if incidence_dir isa Versor
                BoundaryValues(;mesh, densities=φ_interp(φs; k), k)
            else
                map(eachcol(φs)) do φs
                    BoundaryValues(;mesh, densities=φ_interp(φs; k), k)
                end
            end
        end
    end
end

"""
    scattering_boundary(mesh, incidence_dir_iter; kwargs...)

Return a channel of the successive densities for the incidence directions
specified in `incidence_dir_iter`.
"""
scattering_boundary(mesh, incidence_dirs; kwargs...) = Channel{BoundaryValues}(#=1; spawn=false=#) do ch
    with_memoize() do
        fun = scattering_boundary(mesh; kwargs...)
        for idir in incidence_dirs
            put!(ch, fun(idir))
        end
    end
end

binc(mesh, k, dir::Versor, b_builder) = b_builder(incident_field_in_boundary(mesh, k, dir)...; k) 
binc(mesh, k, dirs::AbstractVector, b_builder) = hcat(map(dir -> binc(mesh, k,  dir, b_builder), dirs)...)



function incident_field_in_boundary(mesh, k, idir)
    pinc(x) = cis(k * dot(x, idir))

    pincs    = pinc.(mean.(mesh))
    dn_pincs = ddir.(pinc, mean.(mesh), normal.(mesh))
    
    return pincs, dn_pincs
end





"""
    scattering_nearfield(bv::BoundaryValues)
    scattering_nearfield(mesh::AbstractMesh; kwargs...)

Return a function that computes the scattering on any specified SVector.

`kwargs` are those of `scattering_boundary`.
"""
scattering_nearfield(bv::BoundaryValues) = bv
scattering_nearfield(mesh::AbstractMesh, incidence_dir; kwargs...) = 
    scattering_nearfield(scattering_boundary(mesh, incidence_dir; kwargs...))

"""
    scattering_nearfield(any, pt; kwargs...)

Return the scattering on point `pt`.
"""
scattering_nearfield(any, idir, pt; kwargs...) = scattering_nearfield(any, idir; kwargs...)(pt)

scattering_nearfield(bv::BoundaryValues, pt; kwargs...) = scattering_nearfield(bv)(pt; kwargs...)


"""
    scattering_farfield(bv::BoundaryValues)

Return a function that computes the scattering on any specified SVector.

`kwargs` are those of `scattering_boundary`.
"""
function scattering_farfield(bv::BoundaryValues)
    ff(x::Versor) = bv(x)
    ff(x::SVector) = bv(Versor(x))
    ff(xs) = ff.(xs)
    
    return ff
end


"""
    scattering_farfield(bv, dir; kwargs...)
    
    Return the scattering on the farfield in direction specified by versor `dir`.

    `dir` can also be given as a SVector and the magnitude is ignored.
"""
scattering_farfield(bv::BoundaryValues, pt; kwargs...) = scattering_farfield(bv)(pt; kwargs...)


"""
    scattering_farfield(mesh::AbstractMesh, idir; kwargs...)
"""
function scattering_farfield(mesh::AbstractMesh, incidence_dir, outcedence_dir; kwargs...)
    scattering_farfield(
        scattering_boundary(mesh, incidence_dir; kwargs...),
        outcedence_dir)
end



# Farfield
function (bv::BoundaryValues)(dir::Versor; quad=@quad_gquts(5))
    k = bv.k
    fun(dir) = x -> cis(-k*dot(x, dir))

    return ThreadsX.sum(zip(bv.mesh, bv.densities)) do (el_source, (φL, φM))

        n     = normal(el_source)

        int   = integrate_triangle(fun(dir), el_source; quad)
        
        int/(4*pi) * (φL-im*k*φM*dot(n, dir))
    end
end

# Nearfield
function (bv::BoundaryValues)(x::SVector{3}; quad=@quad_gquts(5))

    # mesh = bv.mesh
    # k    = bv.k
    # dens = bv.densities
    # vels = last.(dens)
    
    # return sum(vel * integrate_triangle( y -> green(x, y; k=k), face; quad) for (face, vel) in zip(mesh, vels))
    φLs = Iterators.map(first, bv.densities)
    φMs = Iterators.map(last, bv.densities)

    return operator(L, bv.mesh, φLs; k=bv.k, quad)(x) + operator(M, bv.mesh, φMs; k=bv.k, quad)(x)
end

(bv::BoundaryValues)(xs; kwargs...) = bv.(xs; kwargs...)


"""
    operator(L, mesh, ψs)
    operator(M, mesh, ψs)

    Return a function of `x` that computes the field of the acoustic monopole (L) or dipole (M) at `x`.
"""
function operator(::Type{L}, mesh::AbstractMesh, φs; k, quad=@quad_gquts(6))
    x -> sum(zip(mesh, φs)) do (face, φ)
        φ * integrate_triangle( y -> green(x, y; k=k), face; quad)
    end
end

function operator(::Type{M}, mesh::AbstractMesh, φs; k, quad=@quad_gquts(6))
    x -> sum(zip(mesh, φs)) do (face, φ)
        ny = normal(face)
        φ * integrate_triangle(face; quad) do y
            ddir(y, ny) do yy
                green(yy, x; k)
            end
        end
    end
end


# If a vector of directions is passed, `solver` needs to work on matrices
# For now, it only supports a single direction at a time
"""
    scattering_boundary_stepwise(mesh, step; kwargs...)

It is functionally equivalent to mapping `scattering_boundary` to submeshes
of `mesh` defined by `step`.

The submeshes always include faces from index 1 to some number.

`step` can be 

- an integer that divides the number of faces of the `mesh` denoting the step in the final face index.
- an iterator of the final face indices.
"""
function scattering_boundary_stepwise(mesh, step; k, incidence_dir::Versor=Versor(0, 0), b_builder, φ_interp, 
    quad=quad_default,
    lm_builder = (mesh, k) -> create_operator_exact_matrix(L, mesh; k, quad), 
    verbose=false,
    kwargs...
    )
    @mytime "Computing boundary" begin
        @mytime "Creating operator" begin
            # Linear map
            A = lm_builder(mesh; k)
        end
        @assert A isa Matrix "The linear map needs to be a matrix"
        @info eltype(A), size(A), "$(Base.summarysize(A)/1e6) MB"

        @mytime false "Computing incident field in boundary" begin
            b = binc(mesh, k, incidence_dir, b_builder)
        end

        @mytime "Solving system" begin
            # Solve A φ = binc
            φs = solve_all_partials(A, b, step; kwargs...)
        end

        return map(φs) do φs
            BoundaryValues(; mesh=mesh[1:length(φs)], densities=φ_interp(φs; k), k) 
        end
    end
end


function solve_all_partials(m, b, idx_iter::AbstractVector; check=true, megacheck=false, verbose=false)

    @mytime "LU factoring" begin
        l, u, = lu_mkl_nopivot(m)
    end

    if check
        err = abs.(l * u .- m)
        maxerr = maximum(err)
        maxval = maximum(abs.(m))
        @info "Maximum error" maxerr
        @info "Relative to max value" maxerr / maxval
        if maxerr / maxval > 1e-3
            @error "The LU decomposition kinda failed" maxerr maxval
        end
    end

    y = l \ b
    
    if megacheck
        luposta = u \ y
        laposta = m \ b
        err2 = abs.(luposta .- laposta)
        @info "Maximum error in result" maximum(err2) maximum(abs.(laposta))
    end

    @info "Memory usage" mem()
    GC.gc()
    @info "Memory usage after `gc()`" mem()
    @info "Solving..."

    @mytime "Solving all Ly=b" begin
        @progress out = [ UpperTriangular(view(u, 1:imax, 1:imax)) \ view(y, 1:imax) for imax in idx_iter ]
    end

    return out
end

function solve_all_partials(m, b, step::Integer; kwargs...)
    n = size(m, 1)
    @assert iszero(mod(n, step)) "The step needs to divide the matrix size"

    return solve_all_partials(m, b, step:step:n; kwargs...)
end


"""
    axis_sweep(axis, n=1000)

Return a vector of versors that rotate around the `axis` which can be `:x`, `:y`, or `:z`.
"""
axis_sweep(axis, n::Integer=100; θmax=2pi) = axis_sweep(axis, range(0; length=n, step=θmax/n))
function axis_sweep(axis, angles::AbstractVector)
    axis == :z && return [ Versor( phi,  pi/2)  for phi   in angles]
    axis == :x && return [ Versor( pi/2, theta) for theta in range(0; length=n, step=θmax/n) ]
    axis == :y && return [ Versor( 0,    theta) for theta in range(0; length=n, step=θmax/n) ]
end
"Rotation matrix around the axis :x, :y, or :z, by angle θ (radians)"
function rotmat(s::Symbol, θ)
    s == :x && return @SMatrix [ 1 0 0; 0 cos(θ) -sin(θ); 0 sin(θ) cos(θ) ]
    s == :y && return @SMatrix [ cos(θ) 0 sin(θ); 0 1 0; -sin(θ) 0 cos(θ) ]
    return @SMatrix [ cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1]
end

function logmessage(n, error)
    # current time
    time = Dates.format(now(UTC), dateformat"yyyy-mm-dd HH:MM:SS")

    # memory the process is using 
    maxrss = mem()

    logdata = (; 
        n, # iteration n
        error, # some super important progress update
        maxrss) # lastly the amount of memory being used

    println(savename(time, logdata; connector=" | ", equals=" = ", sort=false, digits=2))
end


#=
function backscattering(mesh, k, dir=Versor(0,0); b_builder, φ_interp,
    quad=quad_default,
    lm_builder = (mesh, k) -> create_operator_exact_matrix(L, mesh; k, quad), 
    solver= (\)
    )

    @timeit to "Creating operator" begin
        # Linear map
        A = lm_builder(mesh; k)
    end

    # Incident field
    function binc_idir(idir)
        pinc(x) = cis(k * dot(x, idir))

        pincs    = pinc.(mean.(mesh))
        dn_pincs = ddir.(pinc, mean.(mesh), normal.(mesh))

        b_builder(pincs, dn_pincs; k)
    end

    @timeit to "Computing incident field in boundary" begin
        binc = binc_idir(dir)
    end

    @timeit to "Solving system" begin
        # Solve A φ = binc
        φs = solver(A, binc)
    end
    
    
    @timeit to "Computing farfield" begin
        # Compute farfield
        ts = 20log10(abs(
                scattering_farfield(
                    k, 
                    mesh, 
                    φ_interp(φs; k), 
                    -dir;
                    quad
                )))
    end
    

    return ts
    
end



function backscatterings(mesh, k, dirs; b_builder, φ_interp, 
    quad=quad_default,
    lm_builder = (mesh, k) -> create_operator_exact_matrix(L, mesh; k, quad), 
    solver= (\)
    )

    # Linear map
    A = lm_builder(mesh; k)

    # Incident field
    function binc_idir(idir)
        pinc(x) = cis(k * dot(x, idir))

        pincs    = pinc.(mean.(mesh))
        dn_pincs = ddir.(pinc, mean.(mesh), normal.(mesh))

        b_builder(pincs, dn_pincs; k)
    end


    bincs = hcat(binc_idir.(dirs)...)
    

    # Solve A φ = binc
    println("Solving")
    φmat = solver(A, bincs)	
    
    # Compute farfield
    ts = [
        20log10(abs(
            scattering_farfield(
                k, 
                mesh, 
                φ_interp(φs; k), 
                -dir;
                quad
            )))
        for (dir, φs) in zip(dirs, eachcol(φmat))
    ]
    

    return ts
    
end
=#


#=
function solve_all_partials_rec(m, b, idx_iter::AbstractVector)
    l, u, = RecursiveFactorization.lu(m, NoPivot())
    y = UnitLowerTriangular(l) \ b

    [ UpperTriangular(view(u, 1:imax, 1:imax)) \ view(y, 1:imax) for imax in idx_iter ]
end

function solve_all_partials_rec(m, b, step::Integer)
    n = size(m, 1)
    @assert iszero(mod(n, step)) "The step needs to divide the matrix size"

    return solve_all_partials_rec(m, b, step:step:n)
end
=#

#=
function backscattering(mesh, k, dirs=Versor(0, 0); b_builder, φ_interp, 
    quad=quad_default,
    lm_builder = (mesh, k) -> create_operator_exact_matrix(L, mesh; k=k, quad), 
    solver= (\))

    @timeit to "Computing backscattering" begin

        # Boundary
        bvs = scattering_boundary(mesh,dirs; k, b_builder, φ_interp, quad, lm_builder, solver)

        # Farfield
        farfield(bv, idir::Versor) = 20log10(abs(scattering_farfield(k, mesh, bv, -idir; quad)))
        farfield(bvs, idirs) = map(farfield, bvs, idirs)

        ts = @timeit to "Computing farfield" farfield(bvs, dirs)

        return ts
    end
end
=#

