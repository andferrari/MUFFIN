using FITSIO
using IUWT
using Wavelets
using Plots
gr()

include("easy_muffinfunc.jl")

params = parameters_init()

params["freq_pl"] = 1:18:256   # fréquences du cube utilisées
params["nitermax"] = 20
params["μ_s"] = 0.15
params["μ_λ"] = 3.0
params["σ"] = 10.0


params["τ"] = compute_primal_step(params)


params["ρ"] = 0.5
x_r, snr_r, cost_r = easy_muffin_dwt(params)

params["ρ"] = 1.0
x, snr, cost = easy_muffin_dwt(params)


using Plots
pyplot()

p1 = plot([cost cost_r], label=["rho = 1" "rho = 0.5"], ylabel = "Primal Cost")
p2 = plot([snr snr_r], label=["rho = 1" "rho = 0.5"], ylabel = "SNR")
#ti = string("sigma =", string(params["σ"]))
plot(p1,p2)
#file_out = string("results/x_init_","$nitermax","_",@sprintf("%01.03f", μ_s),"_",@sprintf("%01.03f", μ_λ),".jld")
#JLD.save(file_out, "snr", snr,"μ_s",μ_s,"μ_λ",μ_λ)
