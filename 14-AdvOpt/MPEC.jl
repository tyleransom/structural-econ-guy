using JuMP, Ipopt

function cournot_symm(; N = 7, mc = 20, a = 100, b = 2, L = 1000)
    c = mc * ones(N)
    m = Model(Ipopt.Optimizer)
    
    # Suppress IPOPT output (optional)
    set_silent(m)
    
    @variable(m, 0 <= x <= L)  # firm 1 output
    @variable(m, 0 <= y[1:N-1] <= L)  # other firms' output with bounds
    @variable(m, Q >= 0)       # total market output
    
    @constraint(m, Q == x + sum(y[i] for i in 1:N-1))
    @constraint(m, x == y[1])  # symmetry constraint
    
    # Firm 1's objective (negative profit to minimize)
    @NLobjective(m, Min, c[1]*x - x*(a - b*Q))
    
    # Other firms' FOCs as KKT conditions
    # ∂π_i/∂y_i = 0 when 0 < y_i < L
    # This means: (a - b*Q) - b*y_i - c[i+1] = 0
    @NLconstraint(m, foc[i=1:N-1], 
                   (a - b*Q) - b*y[i] - c[i+1] == 0)
    
    optimize!(m)
    
    @show termination_status(m)
    @show objective_value(m)
    @show value(x)
    @show value.(y)
    @show value(Q)
    @show P = a - b*value(Q)
    
    # Check analytical solution
    q_analytical = (a - mc) / (b * (N + 1))
    @assert isapprox(value(x), q_analytical, atol=1e-4)
    println("✓ Solution matches analytical Cournot-Nash equilibrium\n")
    
    return P
end

# 7 firms, varying capacity levels per firm
P7  = cournot_symm()
P17 = cournot_symm(L=17)

# 15 firms, unlimited capacity per firm
P15 = cournot_symm(N=15)
