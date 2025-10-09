using Random, Optim, Statistics, LinearAlgebra

# Generate true data
Random.seed!(123)
N = 10_000
X = [ones(N) randn(N,2)]
β_true = [2.0, -1.0, 0.5]
σ_true = 1.0
y = X * β_true + σ_true * randn(N)

# Auxiliary model (descriptive stats)
function aux_stats(y, X)
    [mean(y); vec(mean(X, dims=1)); cov(y, X[:,2]); cov(y, X[:,3]); 
     cov(X[:,2], X[:,3]); var(y)]
end

β_data = aux_stats(y, X)

# Indirect inference objective
function ii_objective(θ, β_d, X, D=1_000)
    β0, β1, β2, σ = θ
    aux_sims = zeros(length(β_d), D)
    Random.seed!(1234)  # Fixed seed!
    for d in 1:D
        y_sim = β0 .+ β1.*X[:,2] .+ β2.*X[:,3] .+ σ.*randn(size(X, 1))
        aux_sims[:, d] = aux_stats(y_sim, X)
    end
    β_sim_avg = vec(mean(aux_sims, dims=2))
    diff = β_d - β_sim_avg
    return dot(diff, diff)  # J = diff'*W*diff, W=I
end

# Estimate
θ_init = [X \ y; 0.95] .+ 0.1 .* rand(size(X, 2) .+ 1)
θ_init = rand(4)
println("Initial:   β = $(θ_init[1:end-1]), σ = $(θ_init[end])")
result = optimize(θ -> ii_objective(θ, β_data, X, 10), θ_init, LBFGS(), 
                    Optim.Options(show_trace=true, g_tol = 1e-5, 
                                    f_abstol = 1e-5, x_abstol = 1e-5,
                                    iterations=100))

# Results
θ_hat = result.minimizer
println("True:      β = $β_true, σ = $σ_true")
println("Estimated: β = $(round.(θ_hat[1:end-1], digits=3)), σ = $(round(θ_hat[end], digits=3))")
println("OLS check: β = $(round.(X \ y, digits=3)), σ = $(round(std(y - X*(X\y)), digits=3))")
