function create_operator_exact_matrix_lavia(op::Type{<:Operator}, mesh; k, quad=@quad_gquts 6)
    [ operator_elem_lavia(op, face_field, face_src; k, quad) 
        for face_field in mesh, face_src in mesh ]
end

edquad(quad::QuadratureRule) = [ (x, y, w)[i] for ((x, y), w) in zip(quad.points, quad.weights), i in 1:3]


function operator_elem_lavia(::Type{L}, face_field, face_src; k, quad=quad_default)
    ed_quad = edquad(quad)
    lponel = face_field == face_src  

    Operador_Lk(
        k, 
        Array(mean(face_field)), 
        Array(face_src[1]), 
        Array(face_src[2]), 
        Array(face_src[3]), 
        lponel,
        ed_quad
        )
end



function operator_elem_lavia(::Type{M}, face_field, face_src; k, quad=quad_default)
    ed_quad = edquad(quad) 
    
   Operador_Mk(
       k, 
       Array(mean(face_field)), 
       Array(face_src[1]), 
       Array(face_src[2]), 
       Array(face_src[3]), 
       Array(normal(face_src)),
       ed_quad)
end

function operator_elem_lavia(::Type{Mt}, face_field, face_src; k, quad=quad_default)
    ed_quad = edquad(quad) 
    
    p = mean(face_field)
   Operador_Mkt(
       k, 
       Array(p), 
       Array(normal(face_field)),
       Array(face_src[1]), 
       Array(face_src[2]), 
       Array(face_src[3]),
       ed_quad)
end

function operator_elem_lavia(::Type{N}, face_field, face_src; k, quad=quad_default)
    ed_quad = edquad(quad) 
    
    p = mean(face_field)

   Operador_Nk(
       k, 
       Array(p), 
       Array(normal(face_src)),
       Array(face_src[1]), 
       Array(face_src[2]), 
       Array(face_src[3]),
       Array(normal(face_field)),
       face_field==face_src,
       ed_quad)
end

# Función que da el operador discreto 'Mk' para el número de onda 'K' integrando en el triángulo con
# vértices 'QA', 'QB', 'QC' y normal 'Nq' con respecto al triángulo de centroide 'P'.
# Si 'P' pertenece al triángulo 'Q' el operador da exactamente 0 allí (la normal del triángulo es
# ortogonal al vector R = P - Q ); entonces NO TIENE TÉRMINO EN LA DIAGONAL para triángulos planos.
# La regla de cuadratura es una matriz [ x y w ]:
function Operador_Mk( K::Real, P::Array, QA::Array, QB::Array, QC::Array, Nq::Array, QuadRule::Array )
    Mk = ComplexF64( 0 ) ;
    QBMQA = QB - QA ; #
    QCMQA = QC - QA ; #  QCMQB = QC - QB ; #
    for q = 1 : size( QuadRule )[1]
        Q = Punto_Q( q, QA, QuadRule, QBMQA, QCMQA ) ;
        R = norm( P - Q ) ;
        Mk += QuadRule[ q, 3 ] * Integrando_Mk_robust( K, R ) * - dot( P - Q, Nq )  ; # WFPGR*RNQ ; # FPGR
    end
    return  area( QA, QB, QC ) / ( 4*pi ) * Mk ;
end


# Función que da un integrando mejor comportado para 'exp( im*K*R )/R^3*( im*K*R - 1 )' cuando R es pequeño
# Sirve para regularizar los operadores M_k y M_kt (fuera de la diagonal)
# A diferencia del anterior, hemos multiplicado por un factor 1/R
function Integrando_Mk_robust( k::Real, r::Real )
    z = k * r ;
    if abs( z ) > 1E-3
        return k^3 * exp( im * z ) / z^3 * ( im * z - 1 ) ;
    else
        return k^3 * ( - 1 / z^3 - 1/( 2*z ) - im / 3 + z / 8 + im * z^2 / 30 - z^3 / 144 - im * z^4 / 840 + z^5 / 5760 ) ;
    end
end

# Función que da el operador discreto 'Lk' para el número de onda 'K' en el triángulo con
# vértices 'QA', 'QB', 'QC' y centroide 'P'. Si 'P' pertenece al triángulo 'Q' es 'Lponel' = true,
# de lo contrario es false. La regla de cuadratura es una matriz [ x y w ]:
function Operador_Lk( K::Real, P::Array, QA::Array, QB::Array, QC::Array, Lponel::Bool, QuadRule::Array )
    # WARNING : La QuadRule cambiará según Lponel. Esto debe asegurarse desde fuera
    if Lponel # Hay que regularizar el integrando puesto que P pertenece al triángulo Q
        QBMQA = QB - QA ; QCMQA = QC - QA ; QCMQB = QC - QB ;
        # El integrando será : Lk - L0, entonces pre-sumo L0 a Lk
        Qarea = area( QA, QB, QC ) ;
        Lk = Operador_L0( P, QA, QB, QC, QBMQA, QCMQA, QCMQB ) / Qarea ;
        for q = 1 : size( QuadRule )[1]
            R = norm( P - Punto_Q( q, QA, QuadRule, QBMQA, QCMQA ) ) ;
            @inbounds Lk += QuadRule[ q, 3 ] * Integrando_Lk_regular( K, R ) ; # ( FPG - FPG0 ) ;
        end
        return Qarea / ( 4*pi ) * Lk ;
    else # "No Lponel" : No hay que regularizar
        Lk = zero( ComplexF64 ) ;
        QBMQA = QB - QA ; QCMQA = QC - QA ;
        for q = 1 : size( QuadRule )[1]
            R = norm( P - Punto_Q( q, QA, QuadRule, QBMQA, QCMQA ) ) ;
            @inbounds Lk += QuadRule[ q, 3 ] * Integrando_Lk_robust( K, R ) ; # exp( im * K * R ) / ( K * R ) ; # FPG
        end
        return  area( QA, QB, QC ) / ( 4*pi ) * Lk ;
    end
end

function Operador_L0( P::Array, QA::Array, QB::Array, QC::Array, QBMQA::Array, QCMQA::Array, QCMQB::Array )
    L0 = ComplexF64( 0 ) ;
    RQAP = norm( P - QA ) ;
    RQBP = norm( P - QB ) ;
    RQCP = norm( P - QC ) ;
    AR0 = [ RQAP, RQBP , RQCP ] ;
    ARA = [ RQBP, RQCP, RQAP ] ; # podría ser así : ARA = [ AR0[2], AR0[3], AR0[1] ] ;
    AOPP = [ norm( QBMQA ), norm( QCMQB ), norm( QCMQA ) ] ; # AOPP = [ RQAQB, RQBQC, RQCQA ] ;
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

# Función que da un integrando mejor comportado para '(exp( im*K*R ) - 1 )/R' cuando R es pequeño
# Es para elementos en la diagonal. Integrando regularizado.
function Integrando_Lk_regular( k::Real, r::Real )
    z = im * k * r ;
    if abs( z ) > 1E-5
        return im * k * ( exp( z ) - 1 ) / z ; #  ( exp( z ) - 1 ) / r
    else
        return im * k * ( 1 + z / 2 + z^2 / 6 + z^3 / 24 + z^4 / 120 ) ;
    end
end

# Integrando del operador 'Lk' fuera de la diagonal
function Integrando_Lk_robust( k::Real, r::Real )
    z = k * r ;
    if abs( z ) > 1E-4
        return k * exp( im * z ) /  z ;
    else
        K = BiG( k ) ;
        Z = K * BiG( r ) ;
        return ComplexF64( K * ( cos( Z ) / Z + im * SinC( Z ) ) ) ;
    end
end

# Función que da el operador discreto 'Mkt' para el número de onda 'K' integrando en el triángulo con
# vértices 'QA', 'QB', 'QC' con respecto al triángulo de centroide 'P'. Si 'P' pertenece al triángulo 'Q' el operador da
# exactamente 0 allí (la normal del triángulo es ortogonal al vector R = P - Q ).
# La regla de cuadratura es una matriz [ x y w ]:
function Operador_Mkt( K::Real, P::Array, Np::Array, QA::Array, QB::Array, QC::Array, QuadRule::Array )
    Mkt = ComplexF64( 0 ) ;
    QBMQA = QB - QA ; #
    QCMQA = QC - QA ; # QCMQB = QC - QB ; #
    for q = 1 : size( QuadRule )[1]
        Q = Punto_Q( q, QA, QuadRule, QBMQA, QCMQA ) ;
        R = norm( P - Q ) ;
        Mkt += QuadRule[ q, 3 ] * Integrando_Mk_robust( K, R ) * dot( P - Q, Np )  ;
    end
    return  area( QA, QB, QC ) / ( 4*pi ) * Mkt ;
end


function Operador_Nk( K::Real, P::Array, Np::Array, QA::Array, QB::Array,
    QC::Array, Nq::Array, Lponel::Bool, QuadRule::Array )
# WARNING : La QuadRule cambiará según Lponel. Esto debe asegurarse desde fuera
    Nk = ComplexF64( 0 ) ;
    QBMQA = QB - QA ;
    QCMQA = QC - QA ;
    QCMQB = QC - QB ;
    Qarea = area( QA, QB, QC ) ;
    DNPNQ = dot( Np, Nq ) ;
    if Lponel # Hay que regularizar el integrando puesto que P pertenece al triángulo Q
        # El integrando será : Nk - N0 - k^2/2*L0, entonces pre-sumo N0 + k^2/2*L0 a Nk
        Nk += Operador_N0_L0( K, P, QA, QB, QC, QBMQA, QCMQA, QCMQB ) / Qarea ;
        for q = 1 : size( QuadRule )[1]
            Q = Punto_Q( q, QA, QuadRule, QBMQA, QCMQA ) ;
            R = norm( P - Q ) ;
            r = ( P - Q ) / R ;
            z = K * R ;
            dotrNprNq = dot( r, Np ) * dot( r, Nq ) ;
            Nk += QuadRule[ q, 3 ] * Integrando_Nk_regular(K, z, dotrNprNq, DNPNQ ) ;
        end
    else # "No Lponel" : No hay que regularizar
        for q = 1 : size( QuadRule )[1]
            Q = Punto_Q( q, QA, QuadRule, QBMQA, QCMQA ) ;
            R = norm( P - Q ) ;
    r = ( P - Q ) / R ;
    z = K * R ;
    dotrNprNq = dot( r, Np ) * dot( r, Nq ) ;
    Nk += QuadRule[ q, 3 ] * Integrando_Nk_robust( K, z, dotrNprNq, DNPNQ ) ;
        end
    end
    return  Qarea / ( 4*pi ) * Nk ;
end


# Función que da un integrando mejor comportado para Nk cuando 'z' = 'K'*'R' es pequeño
# Es para elementos en la diagonal. Integrando regularizado. No se regulariza teniendo en cuenta
# el producto escalar dot(r,Nx) el cual será en esos casos tendiendo a nulo.
# Cuando la norma de z es pequeña se utiliza una aproximación hasta orden 5
function Integrando_Nk_regular( k::Real, z::Real, dotrNprNq::Real, dotNpNq::Real )
    if abs( z ) > 1E-2
        return  ( k^3 / z^3 ) * ( ( exp( im * z ) * ( 1 - im * z ) - 1 ) * dotNpNq +
            ( exp( im * z ) * ( 3 * im * z - 3 + z^2 ) + 3 ) * dotrNprNq - 1 / 2 * z^2 )
    else
        z2 = z*z ;
        z3 = z*z2 ;
        return k^3 * (
            ( ( ( - z2/5760 + 1/144 ) * z3 - z/8 +  1/(2*z) ) +
            im * ( z2 * ( z2/840 - 1/30 ) + 1/3 ) ) * dotNpNq +
            ( ( ( - z2/1152 + 1/48 ) * z3 - z/8 - 1/(2*z) ) +
            im * z2 * ( z2/210 - 1/15 ) ) * dotrNprNq - 1 / 2 * z2 ) ;
    end
end

# Integrando del operador 'Nk' fuera de la diagonal
# Cuando la norma de z es pequeña se utiliza una aproximación hasta orden 5
function Integrando_Nk_robust( k::Real, z::Real, dotrNprNq::Real, dotNpNq::Real )
    if abs( z ) > 1E-2
        return k^3 * exp( im * z ) / z^3 * ( ( 1 - im * z ) * dotNpNq +
                ( 3 * ( im * z - 1 ) + z^2 ) * dotrNprNq ) ;
    else
        z2 = z*z ;
        z3 = z*z2 ;
        return k^3 * (
            ( ( ( - z2/5760 + 1/144 ) * z3 - z/8 +  1/(2*z) + 1/z3 ) +
                im * ( z2 * ( z2/840 - 1/30 ) + 1/3 ) ) * dotNpNq +
            ( ( ( - z2/1152 + 1/48 ) * z3 - z/8 - 1/(2*z) - 3/z3 ) +
                im * z2 * ( z2/210 - 1/15 ) ) * dotrNprNq ) ;
    end
end

# Función que da los valores de los operadores N0 + k^2/2 * L0 para la regularización
function Operador_N0_L0( K::Real, P::Array, QA::Array, QB::Array, QC::Array,
    QBMQA::Array, QCMQA::Array, QCMQB::Array )
    L0 = Float64( 0 ) ;
    N0 = Float64( 0 ) ;
    RQAP = norm( P - QA ) ;
    RQBP = norm( P - QB ) ;
    RQCP = norm( P - QC ) ;
    AR0 = [ RQAP, RQBP , RQCP ] ;
    ARA = [ RQBP, RQCP, RQAP ] ; # podría ser así : ARA = [ AR0[2], AR0[3], AR0[1] ] ;
    AOPP = [ norm( QBMQA ), norm( QCMQB ), norm( QCMQA ) ] ; # AOPP = [ RQAQB, RQBQC, RQCQA ] ;
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
        N0 += ( ( cos(B+A) - cos(B) ) / R0 / sin( B ) ) ;
    end
    return N0 + K^2/2 * L0
end

# Función que da el punto 'q'-esimo de una regla de cuadratura 'QuadRule' para un triángulo
# con vértice principal 'QA' siendo 'QBMQA' y 'QCMQA' los vectores a los otros vértices.
# Esta implementación es la más rápida hasta ahora
function Punto_Q( q, QA, QuadRule, QBMQA, QCMQA )
    return [ QA[ 1 ] + QuadRule[ q, 1 ] * QBMQA[ 1 ] + QuadRule[ q, 2 ] * QCMQA[ 1 ],
        QA[ 2 ] + QuadRule[ q, 1 ] * QBMQA[ 2 ] + QuadRule[ q, 2 ] * QCMQA[ 2 ],
        QA[ 3 ] + QuadRule[ q, 1 ] * QBMQA[ 3 ] + QuadRule[ q, 2 ] * QCMQA[ 3 ] ] ;
end


#=
"""
    area(v1, v2, v3)
    area(face)

Return the area of a triangle.

# Example

```jldoctest
julia> using Mesh

julia> area((1, 2, 3), (4, 5, 6.2), (7, 8, 9))
0.8485281374238061
```
"""
area((v1, v2, v3) :: Face) = area(v1, v2, v3)
function area(v1, v2, v3)

    v12  = v2 .- v1 
    v13  = v3 .- v1
    ev21 = dot(v12, v12)
    ev31 = dot(v13, v13)
    dp   = dot(v12, v13)

    return sqrt(ev21 * ev31 - dp * dp) / 2
    
end
=#