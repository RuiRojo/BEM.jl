abstract type BoundaryCondition end
abstract type Dirichlet <: BoundaryCondition end
abstract type Neumann <: BoundaryCondition end
abstract type Formulation end
abstract type DirectFormulation <: Formulation end
abstract type IndirectFormulation <: Formulation end

Base.@kwdef struct BoundaryValues <: Function
    mesh :: LightMesh
    densities :: Vector{NTuple{2, Complex{Float}}}    
    k :: Float64
end
    

#=

"""
    compute_boundary(bc, incidence, k, operators)

Calculate vectors `phi` and velocity (*totals*) in the boundary.
""" 
function compute_boundary(bc::BoundaryCondition, boundaryIncidence::Tuple, k::Real, ops)
    phi = compute_phi(bc, SOLVER, boundaryIncidence..., k, ops)
    vel = compute_vel(bc, SOLVER, boundaryIncidence..., k, ops)
    
    return BoundaryValues(zip(phi, vel) |> collect)
end # ¿Para qué estaban estas dos implementaciones?

function compute_boundary(bc::BoundaryCondition, boundaryIncidence::Vector, k::Real, ops)
    valIncs = first.(boundaryIncidence)  |> flatten
    derivIncs = last.(boundaryIncidence) |> flatten

    phis = compute_phi(bc, SOLVER, valIncs, derivIncs, k, ops) |> unflatten
    vels = compute_vel(bc, SOLVER, valIncs, derivIncs, k, ops) |> unflatten
return vels
    return map(phis, vels) do phi, vel
        BoundaryValues(zip(phi, vel) |> collect)
    end

end


Base.broadcast(compute_boundary, bc::BoundaryCondition, bv::Vector, k::Real, r) = 
    compute_boundary(bc, bv, k, r)
=#



#=

#---------- Dirichlet

compute_phi(::Dirichlet, solver, vals, derivs, k, ops) = zero.(vals)

compute_vel(::Dirichlet, solver::MatrixBoundarySolver, 
    vals, derivs, k, ops) = begin
    
    @timeit to "Collecting operators" begin
        @timeit to "Single layer"            lMat  = collect(ops.sl)
        @timeit to "Single layer derivative" mtMat = collect(ops.dn_sl)
    end
   
    aMat = mu(k) .* mtMat + lMat + I/2 # mu(k) .* lMat + mtMat + I/2
    bs   = mu(k) .* derivs + vals # mu(k) .* vals .+ derivs

    @timeit to "Solving" out = matSolve(solver, aMat, bs)

    return out
end



compute_vel(::Dirichlet, solver::GMRES,
    vals, derivs, k, ops) = begin

    
    aLS = mu(k) * ops.dn_sl.lm + ops.sl.lm + I/2
    bs  = mu(k) .* derivs .+ vals
    
    return [ gmres(aLS, bs[:, j]) for j in 1:size(bs, 2)]
end

#--------- Neumann

#----------------
#(\mu N + I/2 - M) p = pi - \mu dn_pi
#----------------
compute_vel(::Neumann, solver, vals, derivs, k, ops) = zero.(derivs)

function compute_phi(::Neumann, solver::MatrixBoundarySolver,
        vals, derivs, k::Real, ops)

    nMat = collect(ops.dn_dl)
    mMat = collect(ops.dl)
    
    aMat = mu(k) .* nMat + I/2 - mMat
    bs   = vals .-  mu(k) .* derivs
    
    return matSolve(solver, aMat, bs)
end

compute_phi(::Neumann, solver::GMRES,
    vals, derivs, k, ops) = begin

    aLS = mu(k) * ops.dn_dl.lm + I/Float(2) - ops.dl.lm
    bs   = vals .-  mu(k) .* derivs
    
    
    
    return [ gmres(aLS, bs[:, j]) for j in 1:size(bs, 2)]
end

# Robin

=#