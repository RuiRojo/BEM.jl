gmres_log = nothing
gmres_logs = []

my_gmres_last = nothing
function my_gmres(A, b::Vector; 
	guess = nothing, # zeros
	reltol_posta=1e-3, 
	restart=min(200, size(A, 2)),
	each_iter=identity,
	maxiter=200, 
	verbose=false,
	very_verbose=false
	)
	
	global gmres_log, my_gmres_last

	local x0 :: Vector{eltype(b)}

	if guess == :last
		if isnothing(my_gmres_last)
			x0 = zeros(eltype(b), size(b))
		else
			x0 = my_gmres_last
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
		x, gmres_log = gmres!(x0, A_ext, b; log=true, initially_zero=iszero(x0), reltol=0, abstol, restart, maxiter, verbose=very_verbose)
		verbose && print('\n')
	end
	verbose && myprintln(gmres_log)
	my_gmres_last = x
	return x
end
function my_gmres(A, bs::Matrix; maxiter=200, reltol=1e-3, restart=min(20, size(A, 2)), verbose=false)
	xs = zeros(eltype(bs), size(bs))
	global gmres_logs = map(eachcol(xs), eachcol(bs)) do x, b
		verbose && print(".")
		gmres!(x, A, b; log=true, reltol, restart, maxiter)[2]
	end

	#global gmres_logs = last.(xlogs)
	return xs #hcat(first.(xlogs)...)
end

#=
function pmy_gmres(A, bs::Matrix)
	xlogs = pmap(eachcol(bs)) do b
		gmres(A, b; log=true, reltol=1e-4)
	end
	global gmres_logs = last.(xlogs)
	return hcat(first.(xlogs)...)
end
=#



function pars(::Type{Dirichlet}, ::Type{DirectFormulation}; 
	combine = true,
	operator_fun=create_operator_exact_matrix,
	solver=(\)
	)
	
	combine == true && (combine = k -> im / (1 + k) )
	
	if combine === L
		(
			b_builder = (p, v; k) -> -p,
			φ_interp  = (φs; k) -> [ (φ, 0) for φ in φs ], 
			lm_builder = (mesh; k) -> operator_fun(L, mesh; k),
			solver
		)
	elseif combine == Mt
		(
		b_builder = (p, v; k) -> -v, 
		φ_interp  = (φs; k) -> [ (φ, 0) for φ in φs ], 
		lm_builder = (mesh; k) -> operator_fun(Mt, mesh; k) + I/2,
		solver
	)
	else
		if combine isa Number
			μ = k -> combine
		else
			@assert combine isa Function
			μ = combine
		end
		(
			b_builder = (p, v; k) -> - p - μ(k) * v , 
			φ_interp  = (φs; k) -> [ (φ, 0) for φ in φs ], 
			lm_builder = (mesh; k) -> 
				operator_fun(L, mesh; k) + μ(k) * (operator_fun(Mt, mesh; k) + I/2),
			solver
		)
	end
end

function pars(::Type{Dirichlet}, ::Type{IndirectFormulation}; 
	combine = true,
	operator_fun=create_operator_exact_matrix,
	solver=(\)
	)
	
	combine == true && (combine = k -> im / (1 + k))

	if combine == L
		(
			b_builder = (p, v; k) -> -p,
			φ_interp  = (φs; k) -> [ (φ, 0) for φ in φs ], 
			lm_builder = (mesh; k) -> operator_fun(L, mesh; k),
			solver
		)
	elseif combine == M
		(
			b_builder  = (p, v; k) -> -p, 
			φ_interp   = (φs; k)   -> [ (0, φ) for φ in φs ], 
			lm_builder = (mesh; k) -> operator_fun(M, mesh; k) + I/2,
			solver
		)
	else
		if combine isa Number
			μ = k -> combine
		else
			@assert combine isa Function
			μ = combine
		end
		(
			b_builder = (p, v; k) -> -p,
			φ_interp  = (φs; k) -> [ (φ, μ(k) * φ ) for φ in φs ], 
			lm_builder = (mesh; k) -> 
				operator_fun(L, mesh; k) + μ(k) * (operator_fun(M, mesh; k) + I/2),
			solver
		)
	end
end


# TMP
dirich_pars_1 = pars(Dirichlet, DirectFormulation; combine=1)
dirich_pars_2 = pars(Dirichlet, DirectFormulation; combine=2)
dirich_pars_3 = pars(Dirichlet, DirectFormulation)

dirich_pars_it_1 = pars(Dirichlet, DirectFormulation; 
						operator_fun=create_operator_exact_lm, 
						solver=(args...) -> my_gmres(args...; maxiter=100),
						combine=1
						)
dirich_pars_it_2 = pars(Dirichlet, DirectFormulation; 
						operator_fun=create_operator_exact_lm, 
						solver=(args...) -> my_gmres(args...; maxiter=100),
						combine=2
						)
dirich_pars_it_3 = pars(Dirichlet, DirectFormulation; 
						operator_fun=create_operator_exact_lm, 
						solver=(args...) -> my_gmres(args...; maxiter=100)
						)

dirich_pars_sit_1 = pars(Dirichlet, DirectFormulation; solver=my_gmres, combine=1)
dirich_pars_sit_2 = pars(Dirichlet, DirectFormulation; solver=my_gmres, combine=2)
dirich_pars_sit_3 = pars(Dirichlet, DirectFormulation; solver=my_gmres)

dirich_pars_it_1_app = pars(Dirichlet, DirectFormulation; 
						solver=my_gmres, 
						combine=1, 
						operator_fun=create_operator_approx
						)


dirich_pars_1 = pars(Dirichlet, DirectFormulation; combine=1)
dirich_pars_2 = pars(Dirichlet, DirectFormulation; combine=2)
dirich_pars_3 = pars(Dirichlet, DirectFormulation)

dirich_pars_it_1 = pars(Dirichlet, DirectFormulation; 
						operator_fun=create_operator_exact_lm, 
						solver=(args...) -> my_gmres(args...; maxiter=100),
						combine=1
						)
dirich_pars_it_2 = pars(Dirichlet, DirectFormulation; 
						operator_fun=create_operator_exact_lm, 
						solver=(args...) -> my_gmres(args...; maxiter=100),
						combine=2
						)
dirich_pars_it_3 = pars(Dirichlet, DirectFormulation; 
						operator_fun=create_operator_exact_lm, 
						solver=(args...) -> my_gmres(args...; maxiter=100)
						)

dirich_pars_sit_1 = pars(Dirichlet, DirectFormulation; solver=my_gmres, combine=1)
dirich_pars_sit_2 = pars(Dirichlet, DirectFormulation; solver=my_gmres, combine=2)
dirich_pars_sit_3 = pars(Dirichlet, DirectFormulation; solver=my_gmres)

dirich_pars_indirect_1 = pars(Dirichlet, IndirectFormulation; combine=1)
dirich_pars_indirect_2 = pars(Dirichlet, IndirectFormulation; combine=2)
dirich_pars_indirect_3 = pars(Dirichlet, IndirectFormulation)

dirich_pars_indirect_3_app = pars(Dirichlet, IndirectFormulation; 
						solver=my_gmres, 
						combine=3, 
						operator_fun=create_operator_approx
						)

dirich_pars_it_1_app = pars(Dirichlet, DirectFormulation; 
						solver=my_gmres, 
						combine=1, 
						operator_fun=create_operator_approx
						)


#=
dirich_pars_1 = (
	b_builder = (p, v; k) -> -p,
	φ_interp  = φs -> [ (φ, 0) for φ in φs ], 
	lm_builder = (mesh; k) -> create_operator_exact_matrix(L, mesh; k, quad=@quad_gquts(6))
)

dirich_pars_it_1 = (
	b_builder = (p, v; k) -> -p, 
	φ_interp  = φs -> [ (φ, 0) for φ in φs ], 
	lm_builder = (mesh; k) -> create_operator_exact_lm(L, mesh; k, quad=@quad_gquts(6)),
	solver = (args...) -> my_gmres(args...; maxiter=100)
)


dirich_pars_it_1_app = (
	b_builder = (p, v; k) -> -p, 
	φ_interp  = φs -> [ (φ, 0) for φ in φs ], 
	lm_builder = (mesh; k) -> create_operator_approx(L, mesh; k, quad=@quad_gquts(6)),
	solver = my_gmres
)

#=
dirich_pars_it_1_app_p = (
	b_builder = (p, v; k) -> p, 
	φ_interp  = φs -> [ (k, vel) for vel in φs ], 
	lm_builder = (mesh; k) -> create_operator_approx(L, mesh; k, quad=@quad_gquts(6)),
	solver = pmy_gmres
)=#

dirich_pars_sit_1 = (
	b_builder = (p, v; k) -> -p, 
	φ_interp  = φs -> [ (φ, 0) for φ in φs ], 
	lm_builder = (mesh; k) -> create_operator_exact_matrix(L, mesh; k, quad=@quad_gquts(6)),
	solver = my_gmres
)


dirich_pars_2 = (
	b_builder = (p, v; k) -> -v, 
	φ_interp  = φs -> [ (φ, 0) for φ in φs ], 
	lm_builder = (mesh; k) -> create_operator_exact_matrix(Mt, mesh; k, quad=@quad_gquts(6)) + I/2	
)

dirich_pars_it_2 = (
	b_builder = (p, v; k) -> -v, 
	φ_interp  = φs -> [ (φ, 0) for φ in φs ], 
	lm_builder = (mesh; k) -> create_operator_exact_lm(Mt, mesh; k, quad=@quad_gquts(6)) + I/2,
	solver = (args...) -> my_gmres(args...; maxiter=100)
)
dirich_pars_sit_2 = (
	b_builder = (p, v; k) -> -v, 
	φ_interp  = φs -> [ (φ, 0) for φ in φs ], 
	lm_builder = (mesh; k) -> create_operator_exact_matrix(Mt, mesh; k, quad=@quad_gquts(6)) + I/2,
	solver = gmres
)

dirich_pars_3 = let
	μ(k) = im / (1 + k)
	quad = @quad_gquts 6
	(
		b_builder = (p, v; k) -> -p - μ(k) * v, 
		φ_interp  = φs -> [ (φ, 0) for φ in φs ], 
		lm_builder = (mesh; k) -> 
			μ(k) * create_operator_exact_matrix(L, mesh; k, quad) + 
			create_operator_exact_matrix(
				Mt, mesh; 
				k, 
				quad
			) + I/2
	)
end

dirich_pars_it_3 = let
	μ(k) = im / (1 + k)
	quad = @quad_gquts 6
	(
		b_builder = (p, v; k) -> -p - μ(k) * v, 
		φ_interp  = φs -> [ (φ, 0) for φ in φs ], 
		lm_builder = (mesh; k) -> 
			μ(k) * create_operator_exact_lm(L, mesh; k, quad) +
			create_operator_exact_lm(
				Mt, mesh; 
				k, 
				quad
			) + I/2,
	solver = (args...) -> my_gmres(args...; maxiter=100)
	)
end


dirich_pars_sit_3 = let
	μ(k) = im / (1 + k)
	quad = @quad_gquts 6
	(
		b_builder = (p, v; k) -> -p - μ(k) * v, 
		φ_interp  = φs -> [ (φ, 0) for φ in φs ], 
		lm_builder = (mesh; k) -> 
			μ(k) * create_operator_exact_matrix(L, mesh; k, quad) + 
			create_operator_exact_matrix(
				Mt, mesh; 
				k, 
				quad
			) + I/2,
	solver = gmres
	)
end

=#

## Neumann

function pars(::Type{Neumann}, ::Type{DirectFormulation}; 
	combine = true,
	operator_fun=create_operator_exact_matrix,
	solver=(\)
	)

	combine == true && (combine = k -> im / (1 + k))
	
	if combine === M
		(
			b_builder = (p, v; k) -> -p,
			φ_interp  = (φs; k) -> [ (0, φ) for φ in φs ], 
			lm_builder = (mesh; k) -> operator_fun(M, mesh; k) - I/2,
			solver
		)
	elseif combine == N
		(
		b_builder = (p, v; k) -> -v, 
		φ_interp  = (φs; k) -> [ (0, φ) for φ in φs ], 
		lm_builder = (mesh; k) -> operator_fun(N, mesh; k),
		solver
	)
	else
		if combine isa Number
			μ = k -> combine
		else
			@assert combine isa Function
			μ = combine
		end
		(
			b_builder = (p, v; k) -> -p - μ(k) * v, 
			φ_interp  = (φs; k) -> [ (0, φ) for φ in φs ], 
			lm_builder = (mesh; k) -> 
				μ(k) * operator_fun(N, mesh; k) + operator_fun(M, mesh; k) - I/2,
			solver
		)
	end
end

function pars(::Type{Neumann}, ::Type{IndirectFormulation}; 
	combine = true,
	operator_fun=create_operator_exact_matrix,
	solver=(\)
	)
	
	combine == true && (combine = k -> im / (1 + k))

	if combine === Mt
		(
			b_builder = (p, v; k) -> -v,
			φ_interp  = (φs; k) -> [ (φ, 0) for φ in φs ], 
			lm_builder = (mesh; k) -> operator_fun(Mt, mesh; k) - I/2,
			solver
		)
	elseif combine == N
		(
		b_builder = (p, v; k) -> -v, 
		φ_interp  = (φs; k) -> [ (0, φ) for φ in φs ], 
		lm_builder = (mesh; k) -> operator_fun(N, mesh; k),
		solver
	)
	else
		if combine isa Number
			μ = k -> combine
		else
			@assert combine isa Function
			μ = combine
		end
		(
			b_builder = (p, v; k) -> -v, 
			φ_interp  = (φs; k) -> [ (φ, μ(k) * φ) for φ in φs ], 
			lm_builder = (mesh; k) -> 
				operator_fun(Mt, mesh; k) - I/2 + μ(k) * operator_fun(N, mesh; k),
			solver
		)
	end
end


# TMP
neum_pars_1 = pars(Neumann, DirectFormulation; combine=1)
neum_pars_2 = pars(Neumann, DirectFormulation; combine=2)
neum_pars_3 = pars(Neumann, DirectFormulation)

neum_pars_it_1 = pars(Neumann, DirectFormulation; 
						operator_fun=create_operator_exact_lm, 
						solver=(args...) -> my_gmres(args...; maxiter=100),
						combine=1
						)
neum_pars_it_2 = pars(Neumann, DirectFormulation; 
						operator_fun=create_operator_exact_lm, 
						solver=(args...) -> my_gmres(args...; maxiter=100),
						combine=2
						)
neum_pars_it_3 = pars(Neumann, DirectFormulation; 
						operator_fun=create_operator_exact_lm, 
						solver=(args...) -> my_gmres(args...; maxiter=100)
						)

neum_pars_sit_1 = pars(Neumann, DirectFormulation; solver=my_gmres, combine=1)
neum_pars_sit_2 = pars(Neumann, DirectFormulation; solver=my_gmres, combine=2)
neum_pars_sit_3 = pars(Neumann, DirectFormulation; solver=my_gmres)

neum_pars_it_1_app = pars(Neumann, DirectFormulation; 
						solver=my_gmres, 
						combine=1, 
						operator_fun=create_operator_approx
						)


neum_pars_1 = pars(Neumann, DirectFormulation; combine=1)
neum_pars_2 = pars(Neumann, DirectFormulation; combine=2)
neum_pars_3 = pars(Neumann, DirectFormulation)

neum_pars_it_1 = pars(Neumann, DirectFormulation; 
						operator_fun=create_operator_exact_lm, 
						solver=(args...) -> my_gmres(args...; maxiter=100),
						combine=1
						)
neum_pars_it_2 = pars(Neumann, DirectFormulation; 
						operator_fun=create_operator_exact_lm, 
						solver=(args...) -> my_gmres(args...; maxiter=100),
						combine=2
						)
neum_pars_it_3 = pars(Neumann, DirectFormulation; 
						operator_fun=create_operator_exact_lm, 
						solver=(args...) -> my_gmres(args...; maxiter=100)
						)

neum_pars_sit_1 = pars(Neumann, DirectFormulation; solver=my_gmres, combine=1)
neum_pars_sit_2 = pars(Neumann, DirectFormulation; solver=my_gmres, combine=2)
neum_pars_sit_3 = pars(Neumann, DirectFormulation; solver=my_gmres)

neum_pars_indirect_1 = pars(Neumann, IndirectFormulation; combine=1)
neum_pars_indirect_2 = pars(Neumann, IndirectFormulation; combine=2)
neum_pars_indirect_3 = pars(Neumann, IndirectFormulation)

#=
neum_pars_1 = (
	b_builder = (p, v; k) -> -p, 
	φ_interp  = φs -> [ (φ, 0) for φ in φs ], 
	lm_builder = (mesh; k) -> create_operator_exact_matrix(M, mesh; k, quad=@quad_gquts 6) - I/2
)

neum_pars_2 = (
	b_builder = (p, v; k) -> -v, 
	φ_interp  = φs -> [ (φ, 0) for φ in φs ], 
	lm_builder = (mesh; k) -> create_operator_exact_matrix(N, mesh; k, quad=@quad_gquts 6)
)


neum_pars_3 = let
	μ(k) = im / (1 + k)
	quad = @quad_gquts 6
	(
		b_builder = (p, v; k) -> -p - μ(k) * v, 
		φ_interp  = φs -> [ (φ, 0) for φ in φs ], 
		lm_builder = (mesh; k) -> (create_operator_exact_matrix(M, mesh; k, quad) - I/2) + μ(k) * create_operator_exact_matrix(N, mesh; k, quad)
	)
end

=#
