version 19.5
clear all
capture log close
log using "HS_sports_example.log", replace

*-------------------------------------------------------------------------------
* Download data from GitHub and load it
*-------------------------------------------------------------------------------
* Download the zip file from GitHub
copy "https://github.com/tyleransom/HS-sports-effects/raw/1.0/NLSY79data/master.dta.zip" "master.dta.zip", replace

* Unzip the file
unzipfile "master.dta.zip", replace

* Load the dataset
use "master.dta", clear

* Clean up the zip file
erase "master.dta.zip"


*-------------------------------------------------------------------------------
* Some additional data cleaning and preparation
*-------------------------------------------------------------------------------
gen teenheightA = height if year==1981  
bys id (year): egen teenheight = mean(teenheightA )
drop teenheightA
gen adultheightA = height if year==1985
bys id (year): egen adultheight = mean(adultheightA )
drop adultheightA
gen teenweightA = weight if year==1981  
bys id (year): egen teenweight = mean(teenweightA )
drop teenweightA
gen q4birth = birthq==4

gen m_rott = mi(rotter_score)
recode rotter_score (. .d .i .r = 0)

gen hgc2   = hgc^2/100

gen exper2 = exper^2/100
gen exper3 = exper^3/1000

gen afqt2  = afqt^2/10
gen rott2  = rotter_score^2/10

gen attCol = inrange(hgc,13,25) if !missInt
capture drop black
gen black  = race==2
* capture drop other
gen other  = race==3 | race==4



*-------------------------------------------------------------------------------
* Program to calibrate rho_k's
*-------------------------------------------------------------------------------
capture program drop calc_rho_k
program define calc_rho_k
    syntax varlist(min=2) [if] [in], Treatment(varname) [NOIsily SORT]
    
    marksample touse
    
    // Separate treatment variable from covariates
    local covars : list varlist - treatment
    
    // Step 1: Run selection regression (Treatment on Controls)
    if "`noisily'" != "" {
        regress `treatment' `covars' if `touse'
    }
    else {
        quietly regress `treatment' `covars' if `touse'
    }
    
    // Step 2: Get full predicted index
    tempvar full_index
    quietly predict `full_index' if `touse', xb
    
    // Step 3: Store results in temporary frame or matrices
    tempname results
    matrix `results' = J(`: word count `covars'', 3, .)
    matrix colnames `results' = coefficient sd_k rho_k
    
    local i = 1
    foreach k of local covars {
        
        // Get coefficient for W_1k
        local pi_k = _b[`k']
        
        // Calculate sqrt(Var(pi_k * W_1k)) - the numerator
        tempvar index_k
        quietly gen `index_k' = `pi_k' * `k' if `touse'
        quietly summarize `index_k' if `touse'
        local sd_k = sqrt(r(Var))
        
        // Calculate sqrt(Var(pi_-k' * W_-k)) - the denominator
        tempvar index_minus_k
        quietly gen `index_minus_k' = `full_index' - `pi_k' * `k' if `touse'
        quietly summarize `index_minus_k' if `touse'
        local sd_minus_k = sqrt(r(Var))
        
        // Calculate rho_k
        local rho_k = `sd_k' / `sd_minus_k'
        
        // Store in matrix
        matrix `results'[`i', 1] = `pi_k'
        matrix `results'[`i', 2] = `sd_k'
        matrix `results'[`i', 3] = `rho_k'
        
        // Clean up temporary variables
        drop `index_k' `index_minus_k'
        
        local i = `i' + 1
    }
    
    // Step 4: Create dataset with results and optionally sort
    preserve
    clear
    svmat `results', names(col)
    gen varname = ""
    local i = 1
    foreach k of local covars {
        replace varname = "`k'" if _n == `i'
        local i = `i' + 1
    }
    
    if "`sort'" != "" {
        gsort -rho_k
    }
    
    // Step 5: Display results
    di ""
    di as text "{hline 70}"
    di as text "Rho_k Calculations: Relative Importance of Each Covariate"
    if "`sort'" != "" {
        di as text "(sorted by rho_k in descending order)"
    }
    di as text "{hline 70}"
    di as text "Covariate" _col(30) "Coefficient" _col(45) "Rho_k"
    di as text "{hline 70}"
    
    forvalues i = 1/`=_N' {
        local k = varname[`i']
        local pi_k = coefficient[`i']
        local rho_k = rho_k[`i']
        di as text "`k'" _col(30) as result %9.4f `pi_k' _col(45) as result %9.4f `rho_k'
    }
    
    di as text "{hline 70}"
    di as text "Note: Rho_k measures importance of variable k relative to all other"
    di as text "      observed covariates in the selection equation."
    di as text "      Compare breakdown point r_X^bp against these rho_k values."
    di as text "{hline 70}"
    di ""
    
    restore
    
end


*-------------------------------------------------------------------------------
* Analysis: HS graduation, college attendance, college graduation by age 25
*-------------------------------------------------------------------------------
* basic regressions
local covars black other hgcMoth hgcFath m_hgcMoth m_hgcFath ///
        lnfamInc78 m_famInc1978 afqt afqt2 m_afqt rotter_score rott2 m_rott ///
        liveWithMom14 femaleHeadHH14 ///
        born1958 born1959 born1960 born1961 born1962 born1963 born1964
regress gradHS athlete          if age==25 & !female
regress gradHS athlete `covars' if age==25 & !female

* calibrating rho_k's
reg athlete `covars' if age==25 & !female
calc_rho_k athlete `covars' if age==25 & !female, t(athlete) noisily sort

* regsensitivity
regsensitivity breakdown gradHS athlete `covars' if age==25 & !female, compare(`covars')
regsensitivity breakdown gradHS athlete `covars' if age==25 & !female, compare(`covars') cbar(0(.1)1)
regsensitivity breakdown gradHS athlete `covars' if age==25 & !female, compare(`covars') rybar(=rxbar) cbar(0(.1)1)
regsensitivity plot
regsensitivity bounds    gradHS athlete `covars' if age==25 & !female, compare(`covars') rybar(=rxbar) cbar(1)
regsensitivity bounds    gradHS athlete `covars' if age==25 & !female, compare(`covars')
regsensitivity plot


log close
