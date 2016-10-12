using JLD
using FITSIO
using IUWT

include("easy_muffinfunc.jl")

#num_simu = parse(Int, ARGS[1]) # for loop
μ_λ = 0.0
nb = 8

nitermax = 10
file_in = "m31_3d_small"
n_simus = 3

μ_s = linspace(0.0, 1.0, n_simus)
snr =  Array(Float64, nitermax+1, n_simus)

for k in 1:n_simus
  x, snr[:,k] = easy_muffin(μ_s[k], μ_λ, nb, nitermax, file_in)
end

JLD.save("my_simu.jld", "snr", snr,"μ_s",μ_s,"μ_λ",μ_λ)

#x, snr_c0 = easy_muffin_c0(μ_s, μ_λ, nb, nitermax, file_in)

#file_out = string("results/x_init_","$nitermax","_",@sprintf("%01.03f", μ_s),"_",@sprintf("%01.03f", μ_λ),".jld")
#JLD.save(file_out, "snr", snr,"μ_s",μ_s,"μ_λ",μ_λ)
