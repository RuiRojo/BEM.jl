using SpecialFunctions

#=
function fn(n; e)
    c1 = sqrt(e * (1 - σ)) / (ρ1  * (1 + σ) * (1 - 2σ))
    c2 = sqrt(e * (1 - σ)) / (2ρ1  * (1 + σ))
    k1 = ω / c1
    k2 = ω / c2
    x1 = k1 * a 
    x2 = k2 * a


    x1jn   = Sbessj(n, x1)
    x1jnp  = x1 * Sbessj_p(n, x1)
    x1jnpp = x1 * Sbessj_2p(n, x1)
    x2jn   = Sbessj(n, x2)
    x2jnp  = Sbessj_p(n, x2)
    x2jnpp = Sbessj_2p(n, x2)

    num = x1jnp / (x1jnp  - x1jn) - (2 * (n^2 + n) * x2jn) / (n^2+n-2) * x2jn + x2^2 * x2jnpp
    den  = x1^2 (σ / (1 - 2σ) * x1jn - x1jnpp) / (x1 * x1jnp - x1jn) -  (2 * (n^2 + n) * (x2jn - x2 * x2jnp)) / ((n^2 + n - 2) * x2jn + x2^2 * x2jnpp)
    
    return ρ / ρ1 * x2^2 / 2 * num / den
end
=#


scattering_nearfield_sphere(radius; bc::Type{<:BoundaryCondition}, k, ecs=eps(), minterms=30, poisson_ratio=0.25, ρ1=1) = function (x_fld)
    bcstr = bc == Dirichlet ? "Dir" : "Neu"

    r_fld = norm(x_fld)
    theta_fld = Versor(x_fld).θ
    nearfield_sphere(k, r_fld, theta_fld, radius, bcstr; ecs, minterms, poisson_ratio, ρ1)
end
scattering_nearfield_sphere(radius, pt; kwargs...) = scattering_nearfield_sphere(radius; kwargs...)(pt)

"""
nearfield_sphere( k, r_fld, theta_fld, radius, bc::String; ecs=eps(), minterms=30, poisson_ratio=0.25, ρ1=1)

Función que da la presión de nearfield para una esfera de radio 'a'
en el punto 'r, theta' (siendo theta el ángulo esférico usual)
debida a una onda plana incidente de amplitud 'P0'.
El cálculo se hace para condiciones de contorno 'bc' que pueden
ser "Dir" o "Neu".
Se asume que la incidencia es en "+z".
"""   # Adaptación de código cortesía de Edmundo Lavia
function nearfield_sphere( k::Real, r::Real, theta::Real, a::Real, bc::String, P0::Real = 1; ecs=eps(), minterms=30, poisson_ratio=0.25, ρ1=1)
    k < 0 && return conj(nearfield_sphere(-k, r, theta, a, bc, P0; ecs, minterms, poisson_ratio, ρ1))
    acumulator = ComplexF64( 0 ) ;
    C = Float64(1) ;
    x = k * a ;
    # Se suman `minterms` términos en la suma o bien hasta 
    # que C sea despreciable en aporte
    j = 0 ;
    if bc == "Dur" # "Durichlet" - prueba (no es culpa de Edmundo) de BC de objeto duro, que puede oscilar en su conjunto (a diferencia de Neumann) 
        while ( abs( C ) > ecs  ) || ( j < minterms )
            C = if j != 1
                - Sbessj_p( j, x ) / Sneum_p( j, x ) ;
            else
                f1 = (1 + 4poisson_ratio ) /( ρ1 * (1 + poisson_ratio))
                -(Sbessj(1, x) * f1 - x * Sbessj_p( 1, x ) ) / (Sneum(1, x) * f1 - x * Sneum_p( 1, x ))
            end
            C2 = C * C ;
            acumulator +=  im^j * ( 2 * j + 1 ) * 
                Shankh( j, k * r ) *
                ( ( C - im * C2 ) / ( 1 + C2 ) ) *
                Legendre_Pl( j, cos( theta ) ) ; 			
            j += 1 ;
        end
    elseif bc == "Neu"
        while ( abs( C ) > ecs  ) || ( j < minterms )
            C = - Sbessj_p( j, x ) / Sneum_p( j, x ) ;
            C2 = C * C ;
            acumulator +=  im^j * ( 2 * j + 1 ) * 
                Shankh( j, k * r ) *
                ( ( C - im * C2 ) / ( 1 + C2 ) ) *
                Legendre_Pl( j, cos( theta ) ) ; 			
            j += 1 ;
        end
    elseif bc == "Dir"
        while ( abs( C ) > ecs  ) || ( j < minterms )
            C = - Sbessj( j, x ) / Sneum( j, x ) ;
            C2 = C * C ;
            acumulator += im^j * ( 2 * j + 1 ) * 
                Shankh( j, k * r ) *
                ( ( C - im * C2 ) / ( 1 + C2 ) ) *
                Legendre_Pl( j, cos( theta ) ) ; 
            j += 1 ;
        end		
    else
        println("Error in bc : 'Neu' or 'Dir' or 'Culo' are the options") ;
        return ;
    end	
    
    acumulator *= cis(-pi/2) # Rui
    # Retorna resultado final
    return  P0 * acumulator ;
end

#
# Polinomios de Legendre. (Para independizarnos de GSL)
#
# Para l >= 0 y -1 <= x <= 1 .
function Legendre_Pl( l::Int64, x::Real )
    # Código tomado de la versión C de GSL. Archivo 'legendre_poly.c'
    # Se adaptó eliminando la estimación del error cometido.
    if(l < 0 || x < -1.0 || x > 1.0)
        println("Error de dominio.")
    elseif l == 0
        return Float64( 1 )
    elseif l == 1
        return x ;
    elseif l == 2
        return 0.5*( 3*x*x - 1 ) ;
    elseif x == 1
        return  Float64( 1 )
    elseif x == -1
        if isodd( l )
            return -Float64( 1 )
        else
            return  Float64( 1 )
        end
    elseif (l < 100000 ) 
        # upward recurrence: l P_l = (2l-1) z P_{l-1} - (l-1) P_{l-2}
        p_ellm2 = 1.0 ;   # P_0(x) 
        p_ellm1 = x ;     # P_1(x)
        p_ell = p_ellm1 ; 
        for ell = 2 : l
            p_ell = ( x*(2*ell-1 )*p_ellm1 - (ell-1)*p_ellm2 )/ell;
            p_ellm2 = p_ellm1;
            p_ellm1 = p_ell;
        end
        return p_ell ;
    else
        # Asymptotic expansion. [Olver, p. 473]
        GSL_ROOT4_DBL_EPSILON = 1.2207031250000000e-04 ; # Magic constant
        u  = l + 0.5 ;
        th = acos( x ) ;
        J0 = besselj0( u*th ) ;
        Jm1 = besselj( -1, u*th ) ;
        # B00 = 1/8 (1 - th cot(th) / th^2    pre = sqrt(th/sin(th))
        if( th < GSL_ROOT4_DBL_EPSILON )
            B00 = ( 1.0 + th*th/15.0 )/24.0 ;
            pre = 1.0 + th*th/12.0;
        else
            sin_th = sqrt( 1.0 - x*x ) ;
            cot_th = x/sin_th;
            B00 = 1.0/8.0*(1.0 - th*cot_th)/( th*th ) ;
            pre = sqrt( th/sin_th ) ;
        end
        c1 = th/u*B00 ;
        return pre*( J0 + c1*Jm1 ) ;
    end
end
# Versión para un array de órdenes l
function Legendre_Pl( L::AbstractArray, x::Real )
    return Legendre_Pl.( L, x ) ;
end


function Hank1( nu::T, z ) where {T<:Real} # Para array de argumentos 'z'
    return besselj.( nu, z ) + im * bessely.( nu, z ) ;
end
function Hank1( nu::AbstractArray, z::Number ) # Para array de órdenes 'nu'
    return besselj.( nu, z ) + im * bessely.( nu, z ) ;
end

# ~~~~~~~~~~~~~~~~~ Función esférica de Bessel de first kind j_n(x)
# Implementación en términos de J_i(x) nativa de Julia. Soporta array de variables 'x'
# o de órdenes 'i'
function Sbessj( i, x )
    return sqrt.( pi * ( 1 ./ ( 2*x ) ) ) .* besselj.( i .+ 0.5, x ) ;
end

# Derivada de la función esférica de Bessel de first kind j_i(x)'
# Se implementó: Morse & Feshbach
# hay alternativas:   -Sbessj(i+1,x) + (i/x)*Sbessj(i,x);
#		      Sbessj(i-1,x) - ((i+1)/x)*Sbessj(i,x);
function Sbessj_p( i, x )
    return ( 1 ./ ( 2 * i .+ 1 ) ) .* ( i .* Sbessj.( i .- 1, x ) - ( i .+ 1 ) .* Sbessj.( i .+ 1, x ) ) ;
end


# ~~~~~~~~~~~~~~~~~ Función esférica de Hankel de first kind h_n^(1)(x)
# Implementación en términos de H_i^(1)( x ) mía (en términos de J_i, ya
# que las Hankel nativas de Julia están buggy --son las de Amos Library--)
# Soportan array de variables o de órdenes
function Shankh( i, x )
    return sqrt.( pi*( 1 ./ ( 2*x ) ) ) .* Hank1.( i .+ 0.5, x ) ;
end

# Derivada de la esférica de Hankel (Bessel third kind)
# No estoy seguro de que sea la más conveniente esta. 
# Soportan array de variables o de órdenes
function Shankh_p( i, x )
    return Shankh.( i .- 1, x ) - ( ( i .+ 1 ) ./ x ) .* Shankh.( i, x ) ; 
end

# ~~~~~~~~~~~~~~~~~ Función esférica de Bessel de second kind y_n(x) ( o bien n_n(x) ) [Neumann]
# Con interfaz try & catch para desbordamientos
function Sneum( i , x )
    try sqrt.( pi*( 1 ./ ( 2 * x ) ) ) .* bessely.( i .+ 0.5, x ) catch ;
        println("WARNING: \n Desbordamiento de Sneum(i,x) en:\n i,x : ", i,",",x) ;
        return 0
    end 
end	


# Derivada de la esférica de Neumann (Bessel 2nd kind)
function Sneum_p( i, x )
    return Sneum.( i .- 1, x ) - ( ( i .+ 1 ) ./ x ) .* Sneum.( i, x ) ;
end	

# Derivada segunda de la esférica de Bessel first kind	
# Basada en Morse & Feshbach (f)
# También está la basada en  Morse & Feshbach (f')
# 		-2*Sbessj_p(i,x) + ((i+1)*i/(x*x) - 1)*Sbessj(i,x); 
function Sbessj_2p( i, x )
    - 2 * ( 1 ./ x ) .* Sbessj.( i .- 1, x ) +
    ( ( ( i .+ 1 ) ./ ( x .* x ) ) .* ( i .+ 2 ) .- 1 ) .* Sbessj.( i, x ) ; 
end