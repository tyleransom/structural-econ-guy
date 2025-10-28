using Random, Statistics, LinearAlgebra, DataFrames, DataFramesMeta, CSV, FixedEffectModels, MixedModels

# Set working directory
cd(@__DIR__)

# Read in the data and clean a bit
df = CSV.read("nlswlearn.csv", DataFrame)
dfuse = df[df.ln_wage.!=999,:]

# FE estimation
@show reg(dfuse, @formula(ln_wage ~ 1 + exper*exper + collgrad + race1 + fe(idcode)), Vcov.cluster(:idcode))

# RE estimation
#categorical!(dfuse, :idcode)
@show fm1 = fit(MixedModel, @formula(ln_wage ~ 1 + exper*exper + collgrad + race1 + (1|idcode)), dfuse)
# gives σ²_ε = .092187 and σ²_a = 0.106297

# Add columns to data indicating the signal (S_{it}) and prior/posterior mean/variances
sig_eps = .092187
# Initialize columns
df.signal = zeros(size(df, 1))
df.priorEbelief = zeros(size(df, 1))
df.postrEbelief = zeros(size(df, 1))
df.priorVbelief = fill(0.106297, size(df, 1))
df.postrVbelief = fill(0.106297, size(df, 1))

# Calculate signals
for i in 1:size(df, 1)
    if df.ln_wage[i] != 999
        df.signal[i] = df.ln_wage[i] - coef(fm1)[1] - coef(fm1)[2]*df.exper[i] - 
                       coef(fm1)[3]*df.collgrad[i] - coef(fm1)[4]*df.race1[i] - 
                       coef(fm1)[5]*df.exper[i]^2
    end
end

# Update beliefs
for id in unique(df.idcode)
    id_rows = findall(df.idcode .== id)
    sorted_rows = id_rows[sortperm(df.t[id_rows])]
    
    for i in 1:length(sorted_rows)
        rowt = sorted_rows[i]
        
        if df.ln_wage[rowt] == 999
            df.signal[rowt] = 0
            df.postrEbelief[rowt] = df.priorEbelief[rowt]
            df.postrVbelief[rowt] = df.priorVbelief[rowt]
        else
            df.postrEbelief[rowt] = df.priorEbelief[rowt]*(sig_eps/(sig_eps + df.priorVbelief[rowt])) + 
                                    df.signal[rowt]*(df.priorVbelief[rowt]/(sig_eps + df.priorVbelief[rowt]))
            df.postrVbelief[rowt] = df.priorVbelief[rowt]*(sig_eps/(sig_eps + df.priorVbelief[rowt]))
        end
        
        if i < length(sorted_rows)
            row_next = sorted_rows[i+1]
            df.priorEbelief[row_next] = df.postrEbelief[rowt]
            df.priorVbelief[row_next] = df.postrVbelief[rowt]
        end
    end
end

first30 = df[1:30,[:ln_wage, :idcode, :t, :signal, :priorEbelief, :postrEbelief, :priorVbelief, :postrVbelief]]
println(first30)
