gmres_log = nothing
gmres_logs = []

gmres_last = nothing
function gmres(A, b::Vector; 
	guess = nothing, # zeros
	reltol_posta=1e-3, 
	restart=min(200, size(A, 2)),
	each_iter=identity,
	maxiter=200, 
	verbose=false,
	very_verbose=false
	)
	
	global gmres_log, gmres_last

	local x0 :: Vector{eltype(b)}

	if guess == :last
		if isnothing(gmres_last)
			x0 = zeros(eltype(b), size(b))
		else
			x0 = gmres_last
		end
	elseif isnothing(guess)
		x0 = zeros(eltype(b), size(b))
	else
		x0 = guess
	end

	abstol = reltol_posta * norm(b)

	A_ext = LinearMap(size(A)...) do x
		each_iter(x0)
		A * x
	end

	@mytime "Applying gmres" begin
		verbose && (printmargin(); print(" "))
		x, gmres_log = IterativeSolvers.gmres!(x0, A_ext, b; log=true, initially_zero=iszero(x0), reltol=0, abstol, restart, maxiter, verbose=very_verbose)
		verbose && print('\n')
	end
	verbose && myprintln(gmres_log)
	gmres_last = x
	return x
end
function gmres(A, bs::Matrix; maxiter=200, reltol=1e-3, restart=min(20, size(A, 2)), verbose=false)
	xs = zeros(eltype(bs), size(bs))
	global gmres_logs = map(eachcol(xs), eachcol(bs)) do x, b
		verbose && print(".")
		IterativeSolvers.gmres!(x, A, b; log=true, reltol, restart, maxiter)[2]
	end

	#global gmres_logs = last.(xlogs)
	return xs #hcat(first.(xlogs)...)
end


#=
function pgmres(A, bs::Matrix)
	xlogs = pmap(eachcol(bs)) do b
		gmres(A, b; log=true, reltol=1e-4)
	end
	global gmres_logs = last.(xlogs)
	return hcat(first.(xlogs)...)
end
=#

##==============================================================

function pars(::Type{Dirichlet}, ::Type{IndirectFormulation}, μ::Function)
    (
        b_builder = (p, v; k) -> -p,
        φ_interp  = (φs; k) -> [ (φ, μ(k) * φ ) for φ in φs ], 
        lm_grtor = op -> (mesh; k) -> op(L, mesh; k) + μ(k) * (op(M, mesh; k) + I/2)
    )
end

function pars(::Type{Dirichlet}, ::Type{DirectFormulation}, μ::Function)
	(
		b_builder = (p, v; k) -> - p - μ(k) * v , 
		φ_interp  = (φs; k) -> [ (φ, 0) for φ in φs ], 
		lm_grtor = op -> (mesh; k) -> op(L, mesh; k) + μ(k) * (op(Mt, mesh; k) + I/2)
    )
end


function pars(::Type{Neumann}, ::Type{IndirectFormulation}, μ::Function)
    (
		b_builder = (p, v; k) -> -v, 
		φ_interp  = (φs; k) -> [ (φ, μ(k) * φ) for φ in φs ], 
		lm_grtor = op -> (mesh; k) -> op(Mt, mesh; k) - I/2 + μ(k) * op(N, mesh; k)
    )
end


function pars(::Type{Neumann}, ::Type{DirectFormulation}, μ::Function)
    (
		b_builder = (p, v; k) -> -p - μ(k) * v, 
		φ_interp  = (φs; k) -> [ (0, φ) for φ in φs ], 
		lm_grtor = op -> (mesh; k) -> μ(k) * op(N, mesh; k) + op(M, mesh; k) - I/2
    )
end

# Default μ
pars(BC, formulation) = pars(BC, formulation, nothing)
pars(BC, formulation, ::Nothing) = pars(BC, formulation, default_μ(BC, formulation))
pars(BC, formulation, μ::Number) = pars(BC, formulation, k -> μ)

default_μ(::Type{Dirichlet}, ::Type{IndirectFormulation}) = k -> im / (1 + k)
default_μ(::Type{Neumann},   ::Type{IndirectFormulation}) = k -> im / (1 + k)
default_μ(::Type{Dirichlet}, ::Type{DirectFormulation})   = k -> im / (1 + k)
default_μ(::Type{Neumann},   ::Type{DirectFormulation})   = k -> im / (1 + k)


# Some helpers if you only want to use one operator (though it shouldn't work very well)
function pars(::Type{Dirichlet}, ::Type{L}, μ...)
	(
		b_builder = (p, v; k) -> -p,
		φ_interp  = (φs; k) -> [ (φ, 0) for φ in φs ], 
		lm_grtor  = op -> (mesh; k) -> op(L, mesh; k)
	)
end

function pars(::Type{Dirichlet}, ::Type{M}, μ...)
	(
		b_builder  = (p, v; k) -> -p, 
		φ_interp   = (φs; k)   -> [ (0, φ) for φ in φs ], 
		lm_grtor = op -> (mesh; k) -> op(M, mesh; k) + I/2
	)
end

function pars(::Type{Dirichlet}, ::Type{Mt}, μ...)
	(
		b_builder = (p, v; k) -> -v, 
		φ_interp  = (φs; k) -> [ (φ, 0) for φ in φs ], 
		lm_grtor = op -> (mesh; k) -> op(Mt, mesh; k) + I/2
	)
end

function pars(::Type{Neumann}, ::Type{M}, μ...)
	(
		b_builder = (p, v; k) -> -p,
		φ_interp  = (φs; k) -> [ (0, φ) for φ in φs ], 
		lm_grtor = op -> (mesh; k) -> op(M, mesh; k) - I/2
	)
end

function pars(::Type{Neumann}, ::Type{Mt}, μ...)
	(
		b_builder = (p, v; k) -> -v,
		φ_interp  = (φs; k) -> [ (φ, 0) for φ in φs ], 
		lm_grtor = op -> (mesh; k) -> op(Mt, mesh; k) - I/2
	)
end

function pars(::Type{Neumann}, ::Type{N}, μ...)
	(
		b_builder = (p, v; k) -> -v, 
		φ_interp  = (φs; k) -> [ (0, φ) for φ in φs ], 
		lm_grtor = op -> (mesh; k) -> op(N, mesh; k)
	)
end