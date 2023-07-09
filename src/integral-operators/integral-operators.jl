"Implementations of algorithms to generate linear maps to apply SurfaceIntegralOperators"
abstract type SurfaceIntegralOperatorAlgorithm end

"""
Exact{Matrix} and Exact{LinearMap} represent exact algorithms for the surface integral operators L/M/Mt/N.
The difference is that Exact{Matrix} produces the operators as matrices, with all the memory allocations that implies,
while Exact{LinearMap} produces a linear map that recomputes the elements as needed.

Exact(..) defaults to creating an Exact{Matrix}
"""
Base.@kwdef struct Exact{T<:Union{Matrix, LinearMap}} <: SurfaceIntegralOperatorAlgorithm
    quad :: QuadratureRule = quad_default
    verbose :: Bool = false
    # Maybe add a special quadrature rule for nearby and/or diagonal elements
end
Exact(;kwargs...) = Exact{Matrix}(;kwargs...)


create_operator_exact_lm(OP, mesh; k, quad=quad_default) = Exact{LinearMap}(;quad)(OP, mesh; k)

function (a::Exact{LinearMap})(::Type{OP}, mesh; k) where OP
    quad = a.quad
    function f(σs)
        ThreadsX.collect(
                # Now sum the effect of all sources
                sum( 
                    σ *  operator_element(OP, face_field, face_src; k, quad)
                    for (σ,  face_src) in zip(σs, mesh))
            for face_field in mesh # for each face
        )
    end
    
    return LinearMap{ComplexF64}(f, length(mesh), length(mesh))
end


create_operator_exact_matrix(OP, mesh; k, verbose=false, quad=quad_default) = Exact{Matrix}(;quad, verbose)(OP, mesh; k)
(a::Exact{Matrix})(::Type{OP}, mesh; k) where OP = @mytime a.verbose "Creating exact matrix operator" begin
    quad = a.quad

    n = length(mesh)
    out = Array{MAT_EL_TYPE}(undef, n, n)

    ThreadsX.foreach(CartesianIndices(out)) do ij
        i, j = Tuple(ij)
        out[i, j] = operator_element(OP, mesh[i], mesh[j]; k, quad)
    end

    return out
end

include(joinpath("Accelerated", "accelerated.jl"))

#=
"""
    create_operator_exact_lm(mesh; k, raccelmin=0)

Return a reference `s` operator that can be used as a rough benchmark.

This is not exact because it doesn't integrate with quad rules and this gets really crappy when near.
"""
function create_operator_exact_lm(::Type{L}, mesh; k, quad=quad_default)
    function f(σs)
        ThreadsX.collect(
            #=
            sum( 
                σ * exacttriangleintegration(k, x, ξ_face)
                for (ξ_face, σ) in zip(mesh, σs)
                ) 
            for x in mean.(mesh)
                =#
            #=sum( 
                σs[idx_src] * exacttriangleintegration(k, mean(mesh[idx_field]), mesh[idx_src], false)
                for idx_src in eachindex(mesh)
                    if idx_src != idx_field
                ) + σs[idx_field] * exacttriangleintegration(k, mean(mesh[idx_field]), mesh[idx_field], true) 
            for idx_field in eachindex(mesh)=#
            let 
                x = mean(mesh[idx_field]) # center, aka collocation point

                # Now sum the effect of all sources. The self contribution is incorporated separately
                sum( 
                    σs[idx_src] * integrate_triangle(y -> green(y, x;k), mesh[idx_src]; quad
                    for idx_src in eachindex(mesh)
                        if idx_src != idx_field
                    ) + σs[idx_field] * intreg(k, mesh[idx_field]) 
                end
            for idx_field in eachindex(mesh) # for each face
        )
    end

    return LinearMap{ComplexF64}(f, length(mesh), length(mesh))
end
=#

# New: also made k a kwarg
# Currently it allocates and is slower than the other version, so I'm not using it.
# Also, ever since I changed something, it's just wrong
#=
operator_element_new(op, f1, f2; k, quad=quad_default) = operator_element_new(ComplexF32, op, f1, f2; k, quad)
function operator_element_new(OUTPUT_TYPE, op, face_field, face_src; k, quad=quad_default) :: OUTPUT_TYPE
    fun = kernel(op; k)
    
    face_src === face_field && return operator_element_self(op, face_field; k, quad)

    nsrc = normal(face_src)
    nfld = normal(face_field)

    xcol = mean(face_field)


    return  integrate_triangle(y -> fun(xcol - y, nsrc, nfld), face_src; quad)

end
=#
operator_element_self(::Type{L}, face; k, quad) = intreg(k, face) 
operator_element_self(::Type{M}, face; k, quad) = 0
operator_element_self(::Type{Mt}, face; k, quad) = 0
operator_element_self(::Type{N}, face; k, quad) = Nlponel(k, face, face; quad)

# Old

function operator_element(::Type{L}, face_field, face_src; k, quad=quad_default) :: ComplexF32
    
    face_src === face_field && return operator_element_self(L, face_field; k, quad)

    x = mean(face_field) # center, aka collocation point

    integrate_triangle(y -> green(y, x;k), face_src; quad)
   
end

function operator_element(::Type{M}, face_field, face_src; k, quad=quad_default) :: ComplexF32
    
    face_src === face_field && return operator_element_self(M, face_field; k, quad)

    ny = normal(face_src)
    x = mean(face_field) # center, aka collocation point

    res = integrate_triangle(face_src; quad) do y
            xmy = x - y
            r = norm(xmy)
            val = green(r; k) * (1 - im * k * r) / r^2
            val * dot(ny, xmy)
    end
    
    return res

end

function operator_element(::Type{Mt}, face_field, face_src; k, quad=quad_default) :: ComplexF32
    
    face_src === face_field && return operator_element_self(Mt, face_field; k, quad)

    nx = normal(face_field)
    x = mean(face_field) # center, aka collocation point

    res = integrate_triangle(face_src; quad) do y
            ymx = y - x
            r = norm(ymx)
            val = green(r; k) * (1 - im * k * r) / r^2
            val * dot(nx, ymx)
    end
    
    return res

end

function operator_element(::Type{N}, face_field, face_src; k, quad=quad_default) :: ComplexF32
    
    face_src === face_field && return operator_element_self(N, face_field; k, quad)

    nx = normal(face_field)
    ny = normal(face_src)
    x = mean(face_field) # center, aka collocation point

    res = integrate_triangle(face_src; quad) do y
            xmy = x - y    
            r = norm(xmy)
            kr = k * r
            ikr = im * kr
            k1 = -(3 + (ikr - 3) * ikr)  / r^4 

            val = k1 * dot(ny, xmy) * dot(nx, xmy) + dot(ny, nx) * (1 - ikr) / r^2
        
            green(r; k) * val
        end
        
    return res
end



function operator_element_aut(::Type{M}, face_field, face_src; k, quad=quad_default)
    ny = normal(face_src)

    x = mean(face_field) # center, aka collocation point

    res = integrate_triangle(face_src; quad) do y
            ddir(y, ny) do y
                green(y, x; k)
            end
         end
    return face_src === face_field ? zero(res) : res

end

function operator_element_aut(::Type{Mt}, face_field, face_src; k, quad=quad_default)
    nx = normal(face_field)
    x = mean(face_field) # center, aka collocation point

    res = ddir(x, nx) do x
            integrate_triangle(face_src; quad) do y
                green(y, x; k)
            end
        end

    return face_src === face_field ? zero(res) : res
end


# Test tmp function: expected to fail for N
function operator_element_aut(::Type{N}, face_field, face_src; k, quad=quad_default)
    ny = normal(face_src)
    nx = normal(face_field)

    x = mean(face_field) # center, aka collocation point

    res :: ComplexF64 = ddir(x, nx) do x
                            integrate_triangle(face_src; quad) do y
                                ddir(y, ny) do y
                                    green(y, x; k)
                                end
                            end
                        end


    return face_src == face_field ? Nlponel(k, face_field, face_src; quad) : res
end


# Regularization


function Nlponel(K, face_field, face_src; quad) :: ComplexF64

	QuadRule = edquad(quad)
	
	P = Array(mean(face_field))
	QA, QB, QC = Array.(vertices(face_src))
	Np = Array(normal(face_field))
	Nq = Array(normal(face_src))
	Qarea = area(face_src)
	QBMQA, QCMQA, QCMQB = QB .- QA, QC .- QA, QC .- QB
	DNPNQ = dot(Np, Nq)
	
   # El integrando será : Nk - N0 - k^2/2*L0, entonces pre-sumo N0 + k^2/2*L0 a Nk
	Nk :: ComplexF64 = Operador_N0_L0( K, P, QA, QB, QC, QBMQA, QCMQA, QCMQB ) / Qarea
	
	for q = 1 : size( QuadRule )[1]
		Q = Punto_Q( q, QA, QuadRule, QBMQA, QCMQA ) ;
		R = norm( P - Q ) ;
		r = ( P - Q ) / R ;
	    z = K * R ;
	    dotrNprNq = dot( r, Np ) * dot( r, Nq ) ;
	    Nk += QuadRule[ q, 3 ] * Integrando_Nk_regular(K, z, dotrNprNq, DNPNQ ) ;
	end
	return Nk * Qarea / 4π
end







"""
    intk0(face)

Return the integral over `face` of 1/r for a field point at the center of `face`.
"""
function intk0(face)
    QA, QB, QC = face
    P = mean(face)
    QBMQA = QB - QA ; QCMQA = QC - QA ; QCMQB = QC - QB ;
	L0 = ComplexF64( 0 ) ;
    RQAP = norm( P - QA ) ;
    RQBP = norm( P - QB ) ;
    RQCP = norm( P - QC ) ;
    AR0 = SA[ RQAP, RQBP , RQCP ] ;
    ARA = SA[ RQBP, RQCP, RQAP ] ; # podría ser así : ARA = [ AR0[2], AR0[3], AR0[1] ] ;
    AOPP = SA[ norm( QBMQA ), norm( QCMQB ), norm( QCMQA ) ] ; # AOPP = [ RQAQB, RQBQC, RQCQA ] ;
    for i = 1 : 3
        R0 = AR0[ i ] ;
        RA = ARA[ i ] ;
        OPP = AOPP[ i ] ;
        if R0 < RA
            TEMP = RA ;
            RA = R0 ;
            R0 = TEMP ;
        end # END IF
        A = acos( ( RA * RA + R0 * R0 - OPP * OPP ) / 2 / RA / R0 ) ;
        B = atan( RA * sin( A ) / ( R0 - RA * cos( A ) ) ) ;
        L0 += ( R0 * sin( B ) * ( log( tan( ( B + A ) / 2 ) ) - log( tan( B / 2 ) ) ) ) ;
    end
    return L0
end

"""
    intreg(k, face;  quad=Quad7)

Return the integral over `face` of cis(k * r) / r for a field point at the center of `face`.
"""
function intreg(k, face; quad= @quad_gquts(10))
    x = mean(face)
    face0 = Face(face[1] .- x, face[2] .- x, face[3] .- x)

    # Returns (cis(k * r) - 1) / r
    function green_reg(y)
        r = norm(y)
   
            # An equivalent expression that doesn't run into indeterminations
        return im * k * cis(k / 2 * r) * sinc(k / (2π) * r)   
    end

    return (intk0(face0) + integrate_triangle(green_reg, face0; quad)) / 4π
end



