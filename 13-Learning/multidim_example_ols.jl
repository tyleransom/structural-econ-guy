using Distributions, LinearAlgebra, Random, DataFrames, Statistics

Random.seed!(1234)

#=============================================================================
DATA GENERATION
=============================================================================#
function simulate_data(N, J, T, β_const, β_exper, β_exper2, β_college, Δ, σ_ε)
    data = []
    abilities = []
    
    for i in 1:N
        college = rand() < 0.3
        a_i = rand(MvNormal(zeros(J), Δ))
        push!(abilities, (i=i, a1=a_i[1], a2=a_i[2], a3=a_i[3], a4=a_i[4], a5=a_i[5], a6=a_i[6]))
        
        for t in 1:T
            exper = t - 1
            exper2 = exper^2
            j = rand(1:J)
            
            log_w = (β_const[j] + β_exper[j]*exper + β_exper2[j]*exper2 + 
                     β_college[j]*college + a_i[j] + σ_ε[j]*randn())
            
            push!(data, (i=i, j=j, t=t, exper=exper, exper2=exper2, 
                        college=college, log_w=log_w))
        end
    end
    
    return DataFrame(data), DataFrame(abilities, [:i, :a1, :a2, :a3, :a4, :a5, :a6])
end

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

#=============================================================================
OLS ESTIMATION
=============================================================================#
function estimate_ols(data, N, J)
    β_est = zeros(4, J)
    σ_est = zeros(J)
    resid_by_person = zeros(N, J)
    
    for j in 1:J
        subset = filter(row -> row.j == j, data)
        X = hcat(ones(nrow(subset)), subset.exper, subset.exper2, subset.college)
        y = subset.log_w
        
        β_est[:, j] = X \ y
        resid = y - X * β_est[:, j]
        
        subset.resid = resid
        person_means = combine(groupby(subset, :i), :resid => mean)
        resid_by_person[person_means.i, j] = person_means.resid_mean
        
        # Within-person variance
        within_var = combine(groupby(subset, :i), :resid => var => :var)
        σ_est[j] = sqrt(mean(filter(!isnan, within_var.var)))
    end
    
    Δ_est = cov(resid_by_person)
    Δ_corr_est = cor(resid_by_person)
    
    return β_est, σ_est, Δ_est, Δ_corr_est
end

β_est, σ_est, Δ_est, Δ_corr_est = estimate_ols(data, N, J)

#=============================================================================
RESULTS
=============================================================================#
println("\n" * "="^70)
println("RESULTS")
println("="^70)

println("\nβ estimates:")
for j in 1:J
    println("Occ $j: True = [$(β_const[j]), $(β_exper[j]), $(β_exper2[j]), $(β_college[j])]")
    println("        Est  = ", round.(β_est[:, j], digits=4))
end

println("\nσ_ε:")
println("True: ", round.(σ_ε_true, digits=3))
println("Est:  ", round.(σ_est, digits=3))

println("\nΔ correlation - True:")
println(round.(cor(Matrix(abilities[:, 2:end])), digits=3))
println("\nΔ correlation - Estimated:")
println(round.(Δ_corr_est, digits=3))


#=============================================================================
BELIEF UPDATING
=============================================================================#
function update_beliefs!(data, β_est, σ_est, Δ_est, N, J)
    # Compute all signals first
    for j in 1:J
        data[!, Symbol("signal_$j")] = zeros(nrow(data))
    end
    
    for row in eachrow(data)
        X = [1.0, row.exper, row.exper2, row.college]
        row[Symbol("signal_$(row.j)")] = row.log_w - dot(X, β_est[:, row.j])
    end
    
    # Pre-allocate belief columns
    for j in 1:J
        data[!, Symbol("prior_mean_$j")] = zeros(nrow(data))
        data[!, Symbol("post_mean_$j")] = zeros(nrow(data))
    end
    
    # Sort once
    sort!(data, [:i, :t])
    
    # Group by person
    gdf = groupby(data, :i)
    
    for person in gdf
        V_t = copy(Δ_est)
        E_t = zeros(J)
        
        for idx in 1:nrow(person)
            j = person.j[idx]
            
            # Store prior
            for k in 1:J
                person[idx, Symbol("prior_mean_$k")] = E_t[k]
            end
            
            # Update
            Ω_t = zeros(J, J)
            Ω_t[j, j] = 1 / σ_est[j]^2
            
            S_t = zeros(J)
            S_t[j] = person[idx, Symbol("signal_$j")]
            
            V_inv = inv(V_t)
            V_t = inv(V_inv + Ω_t)
            E_t = V_t * (V_inv * E_t + Ω_t * S_t)
            
            # Store posterior
            for k in 1:J
                person[idx, Symbol("post_mean_$k")] = E_t[k]
            end
        end
    end
    
    return data
end

# Apply to data
@time data = update_beliefs!(data, β_est, σ_est, Δ_est, N, J)

# Show example for first person
println("\nBeliefs for person 1:")
person1 = filter(r -> r.i == 1, data)
sort!(person1, :t)
for row in first(eachrow(person1), 5)
    signals = [row[Symbol("signal_$k")] for k in 1:J]
    prior = [row[Symbol("prior_mean_$k")] for k in 1:J]
    post = [row[Symbol("post_mean_$k")] for k in 1:J]
    println("t=$(row.t), choice=$(row.j)")
    println("  signals: $(round.(signals, digits=3))")
    println("  prior:   $(round.(prior, digits=3))")
    println("  post:    $(round.(post, digits=3))")
end
