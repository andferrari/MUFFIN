using FITSIO
using IUWT
using Wavelets
using Plots
gr()

include("easy_muffinfunc.jl")

params = parameters_init()

params["freq_pl"] = 1:18:256   # fréquences du cube utilisées
params["nitermax"] = 100
params["μ_s"] = 0.15
params["μ_λ"] = 3.0
params["σ"] = 1.0

params["τ"] = compute_primal_step(params)


x, snr, cost = easy_muffin_dwt(params)


using Plots, LaTeXStrings
gr()

p1 = plot(cost,label="cost")
p2 = plot(snr,label="snr")
ti = string("sigma = ", string(params["σ"]))
plot(p1,p2)
ti = string("sigma =", string(params["σ"]))
plot(p1,p2, title = ti )
#file_out = string("results/x_init_","$nitermax","_",@sprintf("%01.03f", μ_s),"_",@sprintf("%01.03f", μ_λ),".jld")
#JLD.save(file_out, "snr", snr,"μ_s",μ_s,"μ_λ",μ_λ)
