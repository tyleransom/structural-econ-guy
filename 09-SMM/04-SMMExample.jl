using Random, Optim, Statistics, LinearAlgebra

# Generate true data
Random.seed!(123)
N = 10_000
X = [ones(N) randn(N,2)]
β_true = [2.0, -1.0, 0.5]
σ_true = 1.0
y = X * β_true + σ_true * randn(N)

# SMM objective
function smm_objective(θ, X, y, D=1_000)
    β, σ = θ[1:end-1], θ[end]
    
    # Data moments
    g_data = vcat(y, var(y))
    
    # Simulated moments
    Random.seed!(1234)  # Fixed seed!
    g_sim = zeros(N+1, D)
    for d = 1:D
        ε = σ * randn(N)
        y_sim = X * β + ε
        g_sim[:, d] = vcat(y_sim, var(y_sim))
    end
    
    # Moment conditions
    err = g_data - mean(g_sim, dims=2)
    return dot(err, err)  # J = err'*W*err, W=I
end

# Estimate
θ_init = [X \ y; 0.5] .+ 0.1 .* rand(size(X, 2) .+ 1)
println("Initial:   β = $(θ_init[1:end-1]), σ = $(θ_init[end])")
result = optimize(θ -> smm_objective(θ, X, y), θ_init, LBFGS(), 
                    Optim.Options(show_trace=true, g_tol = 1e-5, 
                                    f_abstol = 1e-5, x_abstol = 1e-5))

# Results
θ_hat = result.minimizer
println("True:      β = $β_true, σ = $σ_true")
println("Estimated: β = $(θ_hat[1:end-1]), σ = $(θ_hat[end])")
println("OLS check: β = $(X \ y)")