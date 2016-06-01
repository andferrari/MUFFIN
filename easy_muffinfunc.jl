function easy_muffin(μ_s::Float64, μ_λ::Float64, nb::Int, nitermax::Int, file_in::AbstractString; filename_x0 = [])

  σ = 1.0
  τ = 1e-5

  # load psf, dirty, and sky

  println("loading...")

  f = FITS(string(file_in,"_psf.fits"))
  psf = read(f[1])[:,:,:]
  close(f)
  psfadj = flipdim(flipdim(psf,2),1)


  f = FITS(string(file_in,"_dirty.fits"))
  dirty = read(f[1])[:,:,:]
  close(f)

  f = FITS(string(file_in,".fits"))
  sky = read(f[1])[:,:,:]
  close(f)

  println("size psf ",size(psf))
  println("size dirty ",size(dirty))
  println("size sky ",size(sky))

  nfreq = size(dirty,3)
  nxy = size(dirty,1)

  # initialize

  if isempty(filename_x0)
    x = init_dirty_admm(dirty, psf, psfadj, 5e1)
  else
    x = JLD.load(x0,"x")
  end

  u = zeros(nxy, nxy, nfreq, nb)
  v = zeros(nxy, nxy, nfreq)

  # precomputations
  println("precomputations...")

  psfadj_fft = Array(Complex128,nxy,nxy)
  hth_fft = Array(Complex128,nxy,nxy,nfreq)
  fty = Array(Float64,nxy,nxy,nfreq)

  for k in 1:nfreq
    psfadj_fft = fft(psfadj[:,:,k])
    hth_fft[:,:,k] = fft(ifftshift(ifft(psfadj_fft.*fft(psf[:,:,k]))))
    fty[:,:,k] = real(ifftshift(ifft(fft(dirty[:,:,k]).*psfadj_fft)))
  end

  wstu = Array(Float64,nxy,nxy)
  ∇_freq = Array(Float64,nxy,nxy)
  tmp_spat_cal = Array(Float64,nxy,nxy,nb)
  xt = Array(Float64,nxy,nxy,nfreq)

  # free memory
  psf = 0
  psfadj = 0

  # Iterations

  println("iterate...")

  loop = true
  nrj_sky = vecnorm(sky)^2
  snr = Float64[]
  push!(snr,10*log10(nrj_sky/vecnorm(sky-x)^2))
  println(0, " ", snr[1])

  niter = 0
  while loop
    niter +=1

    t = idct(v)

    for freq in 1:nfreq

      # compute iuwt adjoint
      wstu = iuwt_recomp(squeeze(u[:,:,freq,:],3),0)
      # wstu = iuwt_decomp_adj(u[:,:,freq,:], nb)

      # compute gradient
      ∇_freq = real(ifftshift(ifft(fft(x[:,:,freq]).*hth_fft[:,:,freq]))) - fty[:,:,freq]

      # compute xt
      xt[:,:,freq] = max(x[:,:,freq] - τ*(∇_freq + μ_s*wstu + μ_λ*t[:,:,freq]), 0.0 )

      # update u
      tmp_spat_scal = iuwt_decomp(2*xt[:,:,freq] - x[:,:,freq],nb)

      for b in 1:nb
          u[:,:,freq,b] = sat(u[:,:,freq,b] + σ*μ_s*tmp_spat_scal[:,:,b])
      end

    end

    # update v
    v = sat(v + σ*μ_λ*dct(2*xt - x))

    # update x
    x = copy(xt)

    push!(snr, 10*log10(nrj_sky/vecnorm(sky-x)^2))

    println(niter, " ", snr[niter+1])

    if (niter >= nitermax)
        loop = false
    end

  end

  return x, snr

end


function init_dirty_admm(dirty::Array{Float64,3},psf::Array{Float64,3},psfadj::Array{Float64,3},mu)

    result = zeros(Float64,size(dirty))

    nx = size(dirty,1)
    ny = size(dirty,2)
    nfreq = size(dirty,3)
    nxypsf = size(psf,1)
    psfcbe = zeros(Complex128,size(dirty))
    psfpad = zeros(Float64,size(dirty))
    hty = zeros(Float64,size(dirty))
    for z in 1:nfreq
        hty[:,:,z] = real(ifftshift(ifft(fft(dirty[:,:,z]).*fft(psfadj[:,:,z]))));
    end

    for z in 1:nfreq
        psfpad[1:nxypsf,1:nxypsf,z] = psf[:,:,z]
        psfcbe[:,:,z] = 1./(abs(fft(psfpad[:,:,z])).^2+mu);
        result[:,:,z] = real(ifft(psfcbe[:,:,z].*fft(hty[:,:,z])))
    end

    return result
end


function iuwt_decomp_adj(u1,tmp1,scale)

    tmp1[:,:,1,1] = iuwt_decomp(u1[:,:,1,1],1)[:,:,1]
        for k in 2:scale
            tmp1[:,:,1,1] += iuwt_decomp(u1[:,:,1,k],k)[:,:,k]
        end
    return tmp1
end

function sat(x::Array{Float64,3})
    min(abs(x), 1.0).*sign(x)
end

function sat(x::Array{Float64,2})
    min(abs(x), 1.0).*sign(x)
end

function hthx(x::Array{Float64,2},psf_fft::Array{Complex128,2},psf_adj_fft::Array{Complex128,2},)

    planifft*((planfft*x[:,:,freq]).*psf_fft[:,:,freq])
    tmp_spat_2 = real(planifft*((planfft*tmp_spat_2).*psfadj_fft[:,:,freq])) - fty[:,:,freq]

end
