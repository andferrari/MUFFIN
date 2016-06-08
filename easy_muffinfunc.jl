function easy_muffin(μ_s::Float64, μ_λ::Float64, nb::Int, nitermax::Int, file_in::AbstractString; filename_x0 = [])

  σ = 1.0
  τ = 1e-4

  # load psf, dirty, and sky

  println("loading...")

  folder = "/Users/ferrari/.julia/v0.4/MUFFIN_dev/data/tmp/"

  f = FITS(string(folder,file_in,"_psf.fits"))
  psf = read(f[1])
  close(f)
  if length(size(psf))==4; psf = squeeze(psf,4);end
  psfadj = flipdim(flipdim(psf,2),1)


  f = FITS(string(folder,file_in,"_dirty.fits"))
  dirty = read(f[1])
  close(f)
  if length(size(dirty))==4; dirty = squeeze(dirty,4);end


  f = FITS(string(folder,file_in,"_sky.fits"))
  sky = read(f[1])
  close(f)
  if length(size(sky))==4; sky = squeeze(sky,4);end


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
  niter = 0

  nrj_sky = vecnorm(sky)^2
  snr = Float64[]
  push!(snr,10*log10(nrj_sky/vecnorm(sky-x)^2))
  @printf("%03d | ", niter)
  @printf("%02.04e \n", snr[1])

  while loop
    niter +=1

    t = μ_λ*idct(v, 3)

    for freq in 1:nfreq

      # compute iuwt adjoint
      wstu = iuwt_recomp(squeeze(u[:,:,freq,:],3),0)
      #wstu = IUWT.iuwt_decomp_adj(u[:,:,freq,:], nb)

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
    v = sat(v + σ*μ_λ*dct(2*xt - x, 3))

    # update x
    x = copy(xt)

    push!(snr, 10*log10(nrj_sky/vecnorm(sky-x)^2))
    @printf("%03d | ", niter)
    @printf("%02.04e \n", snr[niter + 1])

    if (niter == nitermax)
        loop = false
    end

  end

  return x, snr

end

function easy_muffin_c0(μ_s::Float64, μ_λ::Float64, nb::Int, nitermax::Int, file_in::AbstractString; filename_x0 = [])

  σ = 1.0
  τ = 1e-4

  # load psf, dirty, and sky

  println("loading...")

  folder = "/Users/ferrari/.julia/v0.4/MUFFIN_dev/data/tmp/"

  f = FITS(string(folder,file_in,"_psf.fits"))
  psf = read(f[1])
  close(f)
  if length(size(psf))==4; psf = squeeze(psf,4);end
  psfadj = flipdim(flipdim(psf,2),1)


  f = FITS(string(folder,file_in,"_dirty.fits"))
  dirty = read(f[1])
  close(f)
  if length(size(dirty))==4; dirty = squeeze(dirty,4);end


  f = FITS(string(folder,file_in,"_sky.fits"))
  sky = read(f[1])
  close(f)
  if length(size(sky))==4; sky = squeeze(sky,4);end


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

  u = zeros(nxy, nxy, nfreq, nb+1)
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
  niter = 0

  nrj_sky = vecnorm(sky)^2
  snr = Float64[]
  push!(snr,10*log10(nrj_sky/vecnorm(sky-x)^2))
  @printf("%03d | ", niter)
  @printf("%02.04e \n", snr[1])

  while loop
    niter +=1

    t = idct(v, 3)

    for freq in 1:nfreq

      # compute iuwt adjoint
      wstu = iuwt_recomp(squeeze(u[:,:,freq,1:nb],3),0,c0=u[:,:,freq,nb+1])

      # compute gradient
      ∇_freq = real(ifftshift(ifft(fft(x[:,:,freq]).*hth_fft[:,:,freq]))) - fty[:,:,freq]

      # compute xt
      xt[:,:,freq] = max(x[:,:,freq] - τ*(∇_freq + μ_s*wstu + μ_λ*t[:,:,freq]), 0.0 )

      # update u
      coef, c0 = iuwt_decomp(2*xt[:,:,freq] - x[:,:,freq], nb, store_c0 = true)
      tmp_spat_scal = cat(3, coef, c0)

      size(tmp_spat_scal)
      size(u)
      for b in 1:nb+1
          u[:,:,freq,b] = sat(u[:,:,freq,b] + σ*μ_s*tmp_spat_scal[:,:,b])
      end

    end

    # update v
    v = sat(v + σ*μ_λ*dct(2*xt - x, 3))

    # update x
    x = copy(xt)

    push!(snr, 10*log10(nrj_sky/vecnorm(sky-x)^2))
    @printf("%03d | ", niter)
    @printf("%02.04e \n", snr[niter + 1])

    if (niter == nitermax)
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


function sat(x::Array{Float64,3})
    min(abs(x), 1.0).*sign(x)
end

function sat(x::Array{Float64,2})
    min(abs(x), 1.0).*sign(x)
end
