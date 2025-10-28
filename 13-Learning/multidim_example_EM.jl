using Distributions, LinearAlgebra, Random, DataFrames, Statistics

Random.seed!(123)

#=============================================================================
DATA GENERATION
=============================================================================#
function simulate_data(N, J, T, β_const, β_exper, β_exper2, β_college, Δ, σ_ε)
    """
    Generate wage data with correlated abilities
    y_ijt = X_it'β_j + a_ij + ε_ijt
    where a_i ~ N(0, Δ)
    """
    data = []
    abilities = []
    
    for i in 1:N
        college = rand() < 0.3
        a_i = rand(MvNormal(zeros(J), Δ))
        push!(abilities, (i=i, a1=a_i[1], a2=a_i[2], a3=a_i[3], a4=a_i[4], a5=a_i[5], a6=a_i[6]))
        
        for t in 1:T
            exper = t - 1
            exper2 = exper^2
            j = rand(1:J)  # random occupation choice
            
            log_w = (β_const[j] + β_exper[j]*exper + β_exper2[j]*exper2 + 
                     β_college[j]*college + a_i[j] + σ_ε[j]*randn())
            
            push!(data, (i=i, j=j, t=t, exper=exper, exper2=exper2, 
                        college=college, log_w=log_w))
        end
    end
    
    ability_cols = [:i, [Symbol("a$j") for j in 1:J]...]
    return DataFrame(data), DataFrame(abilities, ability_cols)
end

# True parameters
N = 10_000
J = 6
T = 20

β_const = [2.0, 2.2, 1.9, 2.5, 2.7, 2.6]
β_exper = [0.08, 0.09, 0.07, 0.12, 0.13, 0.11]
β_exper2 = [-0.002, -0.0022, -0.0018, -0.0025, -0.0027, -0.0023]
β_college = [0.15, 0.18, 0.12, 0.45, 0.50, 0.48]

Δ_true = [1.0  0.6  0.5  0.2  0.15 0.15;
          0.6  1.0  0.55 0.18 0.2  0.17;
          0.5  0.55 1.0  0.15 0.17 0.2;
          0.2  0.18 0.15 1.0  0.65 0.6;
          0.15 0.2  0.17 0.65 1.0  0.62;
          0.15 0.17 0.2  0.6  0.62 1.0]

σ_ε_true = [0.35, 0.32, 0.38, 0.25, 0.22, 0.24]

data, abilities = simulate_data(N, J, T, β_const, β_exper, β_exper2, β_college, Δ_true, σ_ε_true)

println("Generated N=$N, J=$J, T=$T")
println("Total obs: ", nrow(data))

#=============================================================================
PREPARE DATA FOR ESTIMATION
=============================================================================#
function prepare_data(data, N, T, J)
    """
    Convert DataFrame to format needed for EM algorithm
    """
    NT = N * T
    
    # Create y matrix (NT x J) with 999 for unobserved
    y = fill(999.0, NT, J)
    for (idx, row) in enumerate(eachrow(data))
        y[idx, row.j] = row.log_w
    end
    
    # Create x tensor (NT x K x J) - same covariates for all occupations
    K = 4  # const, exper, exper2, college
    x = zeros(NT, K, J)
    for j in 1:J
        x[:, 1, j] = ones(NT)
        x[:, 2, j] = data.exper
        x[:, 3, j] = data.exper2
        x[:, 4, j] = data.college
    end
    
    # Choice vector
    Choice = data.j
    
    return y, x, Choice
end

y, x, Choice = prepare_data(data, N, T, J)

#=============================================================================
EM ALGORITHM
=============================================================================#
function corrcov(C::AbstractMatrix)
    """Convert covariance to correlation"""
    sigma = sqrt.(diag(C))
    return C ./ (sigma * sigma')
end

function init_em(N, T, J, Choice, y, x)
    """
    Initialize EM algorithm with OLS by occupation
    """
    S = 1  # single type for now
    
    BigN = zeros(Int64, J)
    β = zeros(size(x, 2), J)
    resid = Array{Float64}[]
    abilsub = Array{Float64}[]
    Resid = zeros(size(y))
    Csum = zeros(N*S, J)
    tresid = zeros(N*S, J)
    
    for j in 1:J
        BigN[j] = sum(Choice .== j)
        flag = y[:, j] .!= 999
        β[:, j] = x[flag, :, j] \ y[flag, j]
        push!(resid, y[flag, j] .- x[flag, :, j] * β[:, j])
        push!(abilsub, y[flag, j] .- x[flag, :, j] * β[:, j])
        Resid[vec(Choice .== j), j] = resid[j]
        Csum[:, j] = (sum(reshape(Choice .== j, (T, N*S)); dims=1))'
        tresid[:, j] = (sum(reshape(Resid[:, j], (T, N*S)); dims=1))'
    end
    
    # Initial ability estimates
    abil = tresid ./ (Csum .+ eps())
    abil = kron(abil, ones(T, 1))
    
    Psi1 = deepcopy(Csum)
    
    # Add back abilities to residuals
    for j in 1:J
        resid[j] .+= abil[vec(Choice .== j), j]
        Resid[vec(Choice .== j), j] = resid[j]
    end
    
    return BigN, β, resid, abilsub, Resid, Csum, tresid, Psi1
end

function em_step(y, x, Choice, β, N, T, S, J, BigN, Δ, σ, Resid, resid, tresid, Psi1, abilsub, Csum)
    """
    Single EM iteration
    E-step: compute posterior ability means and variances
    M-step: update β, Δ, σ
    """
    abil = zeros(N*S, J)
    
    idelta = inv(Δ)
    vtemp2 = zeros(J, J)
    vabil = zeros(N*S, J)
    
    # Compute total residuals by person
    for j in 1:J
        tresid[:, j] = (sum(reshape(Resid[:, j], (T, N*S)); dims=1))'
    end
    
    # E-step: posterior ability distributions
    for i in 1:(S*N)
        psit = Psi1[i, :]
        Psi = zeros(J, J)
        for j in 1:J
            Psi[j, j] = psit[j] / σ[j]
        end
        
        # Posterior variance: (Δ^-1 + Ψ)^-1
        vtemp = inv(idelta .+ Psi)
        
        # Posterior mean
        vectres = zeros(J)
        for j in 1:J
            vectres[j] = tresid[i, j] / σ[j]
        end
        temp = (vtemp * vectres)'
        abil[i, :] = temp
        
        for j in 1:J
            vabil[i, j] = vtemp[j, j]
        end
        
        # Accumulate for Δ update
        vtemp2 .+= (vtemp .+ temp' * temp)
    end
    
    # M-step: update Δ
    Δ = vtemp2 / N
    
    # Expand abilities to NT
    vabilw = deepcopy(vabil)
    Abil = kron(abil, ones(T, 1))
    
    # M-step: update σ
    sigdem = zeros(J)
    for j in 1:J
        abilsub[j] = Abil[vec(Choice .== j), j]
        sigdem[j] = sum((resid[j] .- abilsub[j]).^2)
    end
    
    σ = ((sum(Csum .* vabilw; dims=1))' .+ sigdem) ./ BigN
    
    # M-step: update β (with updated abilities)
    for j in 1:J
        flag = (y[:, j] .!= 999)
        β[:, j] = x[flag, :, j] \ (y[flag, j] .- abilsub[j])
        resid[j] = y[flag, j] .- x[flag, :, j] * β[:, j]
        Resid[vec(Choice .== j), j] = resid[j]
    end
    
    return β, Δ, σ, Resid, resid, tresid, abilsub
end

function estimate_em(y, x, Choice, N, T, J; maxiter=500, tol=1e-4)
    """
    Run EM algorithm until convergence
    """
    S = 1  # single type
    
    # Initialize
    println("Initializing EM algorithm...")
    BigN, β, resid, abilsub, Resid, Csum, tresid, Psi1 = init_em(N, T, J, Choice, y, x)
    
    # Random initial Δ and σ
    Δ = rand(J, J)
    Δ = 0.5 .* (Δ + Δ')
    σ = rand(J)
    
    println("Starting EM iterations...")
    for iter in 1:maxiter
        Δ_old = copy(Δ)
        
        β, Δ, σ, Resid, resid, tresid, abilsub = 
            em_step(y, x, Choice, β, N, T, S, J, BigN, Δ, σ, Resid, resid, tresid, Psi1, abilsub, Csum)
        
        # Check convergence
        diff = maximum(abs.(Δ_old - Δ))
        
        if iter % 10 == 0
            println("Iter $iter: max |ΔΔ| = ", round(diff, digits=6))
        end
        
        if diff < tol
            println("Converged at iteration $iter")
            break
        end
    end
    
    Δ_corr = corrcov(Δ)
    
    return β, σ, Δ, Δ_corr
end

#=============================================================================
ESTIMATION
=============================================================================#
println("\n" * "="^70)
println("ESTIMATING MODEL")
println("="^70)

β_est, σ_est, Δ_est, Δ_corr_est = estimate_em(y, x, Choice, N, T, J)

#=============================================================================
RESULTS
=============================================================================#
println("\n" * "="^70)
println("RESULTS")
println("="^70)

println("\nβ estimates (const, exper, exper², college):")
for j in 1:J
    println("Occ $j: True = [$(β_const[j]), $(β_exper[j]), $(β_exper2[j]), $(β_college[j])]")
    println("        Est  = ", round.(β_est[:, j], digits=4))
end

println("\nσ_ε estimates:")
println("True: ", round.(σ_ε_true, digits=3))
println("Est:  ", round.(sqrt.(σ_est), digits=3))

println("\nΔ (covariance) - True:")
display(round.(Δ_true, digits=3))
println("\nΔ (covariance) - Estimated:")
display(round.(Δ_est, digits=3))

println("\nΔ (correlation) - True:")
Δ_corr_true = corrcov(Δ_true)
display(round.(Δ_corr_true, digits=3))
println("\nΔ (correlation) - Estimated:")
display(round.(Δ_corr_est, digits=3))

println("\nTrue ability correlations (from data):")
abil_corr_true = cor(Matrix(abilities[:, 2:end]))
display(round.(abil_corr_true, digits=3))
