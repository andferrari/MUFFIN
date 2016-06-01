using JLD
using FITSIO
using IUWT

include("easy_muffinfunc.jl")

μ_λ = 0.0 #3.0
μ_s = 0.25
nb = 8
nitermax = 4000
file_in = "m31_3d"

num_simu = parse(Int, ARGS[1])
μ_s = logspace(-4,1,20)[num_simu]

x, snr = easy_muffin(μ_s, μ_λ, nb, nitermax, file_in)

file_out = string("results/x_init_","$nitermax","_",@sprintf("%01.03f", μ_s),"_",@sprintf("%01.03f", μ_λ),".jld")
JLD.save(file_out, "snr", snr,"μ_s",μ_s,"μ_λ",μ_λ)
