
"""
    green(x, y; k, krmin=1e-3)
    green(xy_vec; k)
    green(r; k)

Return the acoustic field of a unit monopole source with wavenumber k.


exp(im * k * r) / (4π * r)

where r == norm(xy_vec) and xy_vec == x - y.

When k * r is very small, it returns zero.
"""
function green(r; k, krmin=1e-9)
    kr = k * r
    kr > krmin && return cis(kr) / (4 * r * π)
    return (im * k * cis(k / 2 * r) * sinc(k / (2π) * r) + 1 / r) / (4π)
end
green(x, y; k, krmin=1e-10) = green(norm(x .- y); k, krmin)
green(x::AbstractVector; kwargs...) = green(norm(x); kwargs...)




#----------------


kernel(::Type{L}; k, rmin=1e-5) = @SMatrix [ function (x)
    r = norm(x)
    out = green(r; k)
    return r < rmin ? zero(out) : out
end for _ in 1:1, _ in 1:1]


kernel(::Type{M}; k, rmin=1e-5) = @SMatrix [

    function (x)
            r = norm(x)
            out = x[i] * green(r; k) * (1 - im * k * r) / r^2
            return r < rmin ? zero(out) : out
        end
        for i in 1:3, _ in 1:1
            
    ]


kernel(::Type{Mt}; k, rmin=1e-5) = (-) .∘ permutedims(kernel(M; k, rmin))

kernel(::Type{N}; k, rmin=1e-5) = @SMatrix [
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


function monopole_p(y, weight=1; k)
    K = kernel(L; k)
    return x -> weight * first(K(x - y))
end
# to-do
function dipole_p(y, w, weight=1; k)
    K = kernel(L; k)
    return x -> weight * K(x - y)
end
function monopole_dp(y, w, weight=1; k)
    K = kernel(L; k)
    return x -> weight * K(x - y)
end
function dipole_dp(y, w, weight=1; k)
    K = kernel(L; k)
    return x -> weight * K(x - y)
end



#=
kernel_v2(::Type{M}; k, rmin=1e-5) = @SMatrix [
        function (x)
            r = norm(x)
            out = der(Δyi -> green(Δyi * ydir, x; k), 0)
            return r < rmin ? zero(out) : out
        end
        for ydir in (@SVector([1, 0, 0]), @SVector([0, 1, 0]), @SVector([0, 0, 1])), _ in 1:1            
    ]

kernel_v2(::Type{Mt}; k, rmin=1e-5) = [
    function (x)
        r = norm(x)
        out = der(Δxi -> green(@SVector([0, 0, 0]), x + Δxi * xdir; k), 0)
        return r < rmin ? zero(out) : out
    end
    for xdir in (@SVector([1, 0, 0]), @SVector([0, 1, 0]), @SVector([0, 0, 1]))
]

function kernel_v2(::Type{N}; k, rmin=1e-5)
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

function kernel_v3(::Type{N}; k, rmin=1e-5)
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