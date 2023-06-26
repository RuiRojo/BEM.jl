
function create_operator_exact_lm(::Type{OP}, mesh; k, quad=quad_default) where OP
    function f(σs)
        ThreadsX.collect(
                # Now sum the effect of all sources
                sum( 
                    σ *  operator_elem(OP, face_field, face_src; k, quad)
                    for (σ,  face_src) in zip(σs, mesh))
            for face_field in mesh # for each face
        )
    end
    
    return LinearMap{ComplexF64}(f, length(mesh), length(mesh))
end

create_operator_exact_matrix(::Type{OP}, mesh; k, verbose=false, quad=quad_default) where OP = @mytime verbose "Creating exact matrix operator" begin

    n = length(mesh)
    out = Array{MAT_EL_TYPE}(undef, n, n)

    ThreadsX.foreach(CartesianIndices(out)) do ij
        i, j = Tuple(ij)
        out[i, j] = operator_elem(OP, mesh[i], mesh[j]; k, quad)
    end

    return out
end


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
operator_elem_new(op, f1, f2; k, quad=quad_default) = operator_elem_new(ComplexF32, op, f1, f2; k, quad)
function operator_elem_new(OUTPUT_TYPE, op, face_field, face_src; k, quad=quad_default) :: OUTPUT_TYPE
    fun = potential(op; k)
    
    face_src === face_field && return operator_elem_self(op, face_field; k, quad)

    nsrc = normal(face_src)
    nfld = normal(face_field)

    xcol = mean(face_field)


    return  integrate_triangle(y -> fun(xcol - y, nsrc, nfld), face_src; quad)

end
=#
operator_elem_self(::Type{L}, face; k, quad) = intreg(k, face) 
operator_elem_self(::Type{M}, face; k, quad) = 0
operator_elem_self(::Type{Mt}, face; k, quad) = 0
operator_elem_self(::Type{N}, face; k, quad) = Nlponel(k, face, face; quad)

# Old

function operator_elem(::Type{L}, face_field, face_src; k, quad=quad_default) :: ComplexF32
    
    face_src === face_field && return operator_elem_self(L, face_field; k, quad)

    x = mean(face_field) # center, aka collocation point

    integrate_triangle(y -> green(y, x;k), face_src; quad)
   
end

function operator_elem(::Type{M}, face_field, face_src; k, quad=quad_default) :: ComplexF32
    
    face_src === face_field && return operator_elem_self(M, face_field; k, quad)

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

function operator_elem(::Type{Mt}, face_field, face_src; k, quad=quad_default) :: ComplexF32
    
    face_src === face_field && return operator_elem_self(Mt, face_field; k, quad)

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

function operator_elem(::Type{N}, face_field, face_src; k, quad=quad_default) :: ComplexF32
    
    face_src === face_field && return operator_elem_self(N, face_field; k, quad)

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



function operator_elem_aut(::Type{M}, face_field, face_src; k, quad=quad_default)
    ny = normal(face_src)

    x = mean(face_field) # center, aka collocation point

    res = integrate_triangle(face_src; quad) do y
            ddir(y, ny) do y
                green(y, x; k)
            end
         end
    return face_src === face_field ? zero(res) : res

end

function operator_elem_aut(::Type{Mt}, face_field, face_src; k, quad=quad_default)
    nx = normal(face_field)
    x = mean(face_field) # center, aka collocation point

    res = ddir(x, nx) do x
            integrate_triangle(face_src; quad) do y
                green(y, x; k)
            end
        end

    return face_src === face_field ? zero(res) : res
end

function operator_elem_aut(::Type{N}, face_field, face_src; k, quad=quad_default)
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

# I think this is only for the self case, so maybe it could receive just a single face?
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


