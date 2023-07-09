
    # Convolver

    # To-do: check if the type instabilities here matter

# I don't think this needed to be a type. 
# It could just have been an operator that builds a closure and wraps it in a LinearMap
# It would have been neater maybe, and saved the implementation of size and others


"""
A Convolver is reseponsible for the convolution step in the acceleration.

"""
struct Convolver#{F, IF} # Check if this stuff is necessary
    FFT! #:: F
    IFFT! #:: IF
    grid  :: Grid
    gf :: Array{ComplexF64, 3} # ComplexF64 because FFTW is way slower for ComplexF32
    σ_grid_pad_pre :: Union{Missing, Array{ComplexF64, 3}} # Pre-allocated space for the padded input array
end


"""
Create a function that takes an array, interprets it as corresponding to values on`grid`,
and performs (linear) convolution with `fun`.
"""
Convolver(func, grid; preallocate=false, ceilfun=(n->nextprod((2, 3), n)), verbose=false) = mytime("Convolver"; verbose) do
    σ_grid_size = size(grid)
    npad = ceilfun.(2 .* σ_grid_size .- 1)
    
    
    σ_grid_pad_pre = preallocate ? zeros(ComplexF64, npad) : missing


    g_arr = let
        Ns = size(grid) 
        coords = (:).( .-Ns .+ 1, npad .- Ns #=Ns .- 1=#) .* step(grid) 
        arr = ComplexF64[ func(SVector(pt)) for pt in Iterators.product(coords...) ]
        circshift(arr, .-Ns .+ 1)
    end

    gf    = fft(g_arr); 
    FFT!  = plan_fft!(gf)#plan_fft!(Array{ComplexF64}(undef, npad)) # Maybe I should convolve with more precision
    IFFT! = MyIFFT!(FFT!) # Because otherwise some times it takes way longer and allocates way more than the FFT, don't know why
         #plan_ifft!(gf)#plan_ifft!(Array{ComplexF64}(undef, npad)) #inv(FFT!)

    return Convolver(FFT!, IFFT!, grid, gf, σ_grid_pad_pre)
end
    # (temporary kludge because, last I checked, the built-in ifft in my PC was slower than this for whatever reason)
    struct MyIFFT!
        FFT!
    end
    function Base.:*(op::MyIFFT!, x)
        out = op.FFT! * myswap3!(x) 
        out ./= length(x)
        return out
    end
    function myswap3!(x)
        reverse!(@view(x[2:end, :, :]); dims=1)
        reverse!(@view(x[:, 2:end, :]); dims=2)
        reverse!(@view(x[:, :, 2:end]); dims=3)
        x
    end


@memoize convolver_buf(ns) = zeros(ComplexF64, ns)
push!(MEMO_FUNS, convolver_buf)

Base.:*(c::Convolver, σ_grid) = @mytime "Convolving" begin
    @assert size(σ_grid) == size(c.grid) "The input grid has size $(size(σ_grid)) but $size(c.grid) was expected"

    σ_grid_pad_pre  = c.σ_grid_pad_pre
    FFT!        = c.FFT!
    IFFT!       = c.IFFT!
    gf          = c.gf

    begin
        σ_grid_pad :: Array{ComplexF64, 3} = if ismissing(σ_grid_pad_pre)
            convolver_buf(size(gf)) # zero(gf)
        else
            σ_grid_pad_pre
        end
        σ_grid_pad .= 0
        σ_grid_pad[CartesianIndices(σ_grid)] .= σ_grid
    end

    out   = let
        @mytime "FFT" FFT! * σ_grid_pad
        σ_grid_pad .*= gf
        @mytime "IFFT" IFFT! * σ_grid_pad
    end

    return ComplexF32.(view(out, CartesianIndices(σ_grid))) # lo trunco al tamaño de q
end
