using Distributions, DataFrames, StatsPlots, Random, Statistics
using Plots, SpecialFunctions

# Ensure GR backend is active (most reliable for VS Code)
gr()

# Set theme for clean aesthetics (similar to theme_minimal)
theme(:default)
default(
    fontfamily = "Computer Modern",
    linewidth = 2,
    framestyle = :box,
    grid = false,
    legend = false,
    guidefontsize = 10,
    tickfontsize = 8,
    titlefontsize = 12
)

# Simulation parameters
N = 100
N_frechet = 10_000
B = 1_000_000
Random.seed!(123)  # For reproducibility

# ============================================================================
# CLT - Mean (or Median) → Normal
# ============================================================================

println("Running CLT simulation...")
exp_dist = Exponential(2.0)  # rate = 0.5 → scale = 1/0.5 = 2.0

# Simulate B replications
mean_stats = [mean(rand(exp_dist, N)) for _ in 1:B]

# Create histogram with normal overlay
μ_fit = mean(mean_stats)
σ_fit = std(mean_stats)
normal_fit = Normal(μ_fit, σ_fit)

p1 = histogram(
    mean_stats,
    bins = 150,
    normalize = :pdf,
    alpha = 0.7,
    color = :lightgray,
    label = "",
    xlabel = "Sample Mean",
    ylabel = "Density",
    title = "CLT: Mean → Normal"
)
plot!(p1, x -> pdf(normal_fit, x), 
      color = :navy, 
      linewidth = 2, 
      label = "")

display(p1)
gui()

# ============================================================================
# EVT - Maximum → Gumbel
# ============================================================================

println("Running EVT (Maximum) simulation...")
max_stats = [maximum(rand(exp_dist, N)) for _ in 1:B]

# Fit Gumbel parameters using method of moments
# μ = mean - γ*σ*√6/π, β = σ*√6/π (γ ≈ 0.5772 is Euler-Mascheroni constant)
γ = 0.5772156649015329  # Euler-Mascheroni constant
μ_gumbel = mean(max_stats) - γ * std(max_stats) * sqrt(6) / π
β_gumbel = std(max_stats) * sqrt(6) / π
gumbel_fit = Gumbel(μ_gumbel, β_gumbel)

p2 = histogram(
    max_stats,
    bins = 150,
    normalize = :pdf,
    alpha = 0.7,
    color = :lightgray,
    label = "",
    xlabel = "Sample Maximum",
    ylabel = "Density",
    title = "EVT: Maximum → Gumbel"
)
plot!(p2, x -> pdf(gumbel_fit, x), 
      color = :navy, 
      linewidth = 2, 
      label = "")

display(p2)
gui()


# ============================================================================
# EVT - Maximum (Heavy Tails) → Fréchet
# ============================================================================

println("Running EVT (Fréchet) simulation...")
α_true = 1.5

# Generate Pareto draws
pareto_dist = Pareto(α_true, 1.0)  

# Take maximum with larger sample
max_stats_pareto = [maximum(rand(pareto_dist, N_frechet)) for _ in 1:B]

# Normalize: for large N, M_N / N^(1/α) → standard Fréchet
a_N = N_frechet^(1/α_true)
normalized_max = max_stats_pareto ./ a_N

println("Mean of normalized maxima: $(mean(normalized_max))")
println("Median of normalized maxima: $(median(normalized_max))")
println("99th percentile: $(quantile(normalized_max, 0.99))")
println("Max value: $(maximum(normalized_max))")

# Filter to reasonable range for visualization (keep 99.5% of data)
upper_limit = quantile(normalized_max, 0.995)
filtered_max = normalized_max[normalized_max .<= upper_limit]

println("Plotting up to $(round(upper_limit, digits=2))")

# Use Distributions.jl directly
frechet_fit = Frechet(α_true, 1.0)  # shape = 1.5, scale = 1

p3 = histogram(
    filtered_max,
    bins = 100,
    normalize = :pdf,
    alpha = 0.7,
    color = :lightgray,
    label = "",
    xlabel = "Sample Maximum",
    ylabel = "Density",
    title = "EVT: Maximum (Heavy Tails) → Fréchet"
)

# Use pdf() directly from Distributions.jl
plot!(p3, x -> pdf(frechet_fit, x),
      color = :navy, 
      linewidth = 3,
      xlim = (0, upper_limit))

display(p3)
gui()


# ============================================================================
# EVT - Minimum → Exponential
# ============================================================================

println("Running EVT (Minimum) simulation...")
min_stats = [minimum(rand(exp_dist, N)) for _ in 1:B]

# Fit exponential to the minimum
rate_fit = 1 / mean(min_stats)
exp_fit = Exponential(1 / rate_fit)  # Exponential uses scale parameter

p4 = histogram(
    min_stats,
    bins = 150,
    normalize = :pdf,
    alpha = 0.7,
    color = :lightgray,
    label = "",
    xlabel = "Sample Minimum",
    ylabel = "Density",
    title = "EVT: Minimum → Exponential"
)
plot!(p4, x -> pdf(exp_fit, x), 
      color = :navy, 
      linewidth = 2, 
      label = "")

display(p4)
gui()

println("Simulations complete!")