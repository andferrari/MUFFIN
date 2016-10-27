myfft(x) = fft(x,(1:2))
myifft(x) = ifft(x,(1:2))
myifftshift(x) = ifftshift(ifftshift(x,1),2)

function easy_muffin(μ_s::Float64, μ_λ::Float64, nb::Int, nitermax::Int, file_in::AbstractString; filename_x0 = [])

  σ = 1.0
  τ = 1e-4


  # load psf, dirty, and sky

  println("loading...")

  folder = "/Users/ferrari/.julia/v0.4/MUFFIN_dev/data/"
  folder="/Users/antonyschutz/Desktop/MUFFIN_dev.jl-master/data/"
  println(string(folder,file_in,"_psf.fits"))
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

  psfadj_fft = myfft(psfadj)
  hth_fft = myfft(myifftshift(myifft(psfadj_fft.*myfft(psf))))
  fty = real(myifftshift(myifft(myfft(dirty).*psfadj_fft)))

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
    # compute gradient
    ∇_freq = real(myifftshift(myifft(myfft(x).*hth_fft))) - fty
    t = idct(v, 3)

    for freq in 1:nfreq

      # compute iuwt adjoint
      wstu = iuwt_recomp(squeeze(u[:,:,freq,:],3),0)
      #wstu = IUWT.iuwt_decomp_adj(u[:,:,freq,:], nb)

      # compute xt
      xt[:,:,freq] = max(x[:,:,freq] - τ*(∇_freq[:,:,freq] + μ_s*wstu + μ_λ*t[:,:,freq]), 0.0 )

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


function easy_muffin_dwt(τ::Float64, μ_s::Float64, μ_λ::Float64, nitermax::Int, file_in::AbstractString; filename_x0 = [])

  σ = 1.0
  spatialwlt  = [WT.db1,WT.db2,WT.db3,WT.db4,WT.db5,WT.db6,WT.db7,WT.db8]
  nb = length(spatialwlt)

  # load psf, dirty, and sky

  println("loading...")

  folder = "/Users/ferrari/.julia/v0.4/MUFFIN_dev/data/"

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
  psf_fft = Array(Complex128,nxy,nxy,nfreq)

  for k in 1:nfreq
    psfadj_fft = fft(psfadj[:,:,k])
    psf_fft[:,:,k] = fft(psf[:,:,k])
    hth_fft[:,:,k] = fft(ifftshift(ifft(psfadj_fft.*psf_fft[:,:,k])))
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

  cost = Float64[]
  tmp = 0.0
  for freq in 1:nfreq
    tmp = tmp + 0.5vecnorm(dirty[:,:,freq] -  ifftshift(ifft(fft(x[:,:,freq]).*psf_fft[:,:,freq])))^2
    for b in 1:nb
      tmp = tmp + μ_s*sumabs(dwt(x[:,:,freq],wavelet(spatialwlt[b])))
    end
  end
  push!(cost,tmp)

  @printf("%03d | ", niter)
  @printf("%02.04e \n", snr[1])

  while loop
    niter +=1

    t = idct(v, 3)

    for freq in 1:nfreq

      # compute iuwt adjoint

      #wstu = iuwt_recomp(squeeze(u[:,:,freq,:],3),0)
      wstu = idwt(u[:,:,freq,1],wavelet(spatialwlt[1]))
      for b in 2:nb
        wstu = wstu + idwt(u[:,:,freq,b],wavelet(spatialwlt[b]))
      end

      #wstu = IUWT.iuwt_decomp_adj(u[:,:,freq,:], nb)

      # compute gradient
      ∇_freq = real(ifftshift(ifft(fft(x[:,:,freq]).*hth_fft[:,:,freq]))) - fty[:,:,freq]

      # compute xt
      xt[:,:,freq] = max(x[:,:,freq] - τ*(∇_freq + μ_s*wstu + μ_λ*t[:,:,freq]), 0.0 )

      # update u

      tmp_spat_cal[:,:,1] = dwt(2*xt[:,:,freq] - x[:,:,freq],wavelet(spatialwlt[1]))
      for b in 2:nb
          tmp_spat_cal[:,:,b] = dwt(2*xt[:,:,freq] - x[:,:,freq],wavelet(spatialwlt[b]))
      end

      for b in 1:nb
          u[:,:,freq,b] = sat(u[:,:,freq,b] + σ*μ_s*tmp_spat_cal[:,:,b])
      end

    end

    # update v
    v = sat(v + σ*μ_λ*dct(2*xt - x, 3))

    # update x
    x = copy(xt)

    push!(snr, 10*log10(nrj_sky/vecnorm(sky-x)^2))
    tmp = 0.0
    for freq in 1:nfreq
      tmp = tmp + 0.5vecnorm(dirty[:,:,freq] -  ifftshift(ifft(fft(x[:,:,freq]).*psf_fft[:,:,freq])))^2
      for b in 1:nb
        tmp = tmp + μ_s*sumabs(dwt(x[:,:,freq],wavelet(spatialwlt[b])))
      end
    end
    push!(cost,tmp)
    @printf("%03d | ", niter)
    @printf("%02.04e | ", snr[niter + 1])
    @printf("%02.04e \n", cost[niter + 1])
    if (niter == nitermax)
        loop = false
    end

  end

  return x, snr, cost

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

    hty = real(myifftshift(myifft(myfft(dirty).*myfft(psfadj))));
    psfcbe = 1./(abs(myfft(psf)).^2+mu);
    result = real(myifft(psfcbe.*myfft(hty)))

    return result
end


function sat(x::Array{Float64,3})
    min(abs(x), 1.0).*sign(x)
end

function sat(x::Array{Float64,2})
    min(abs(x), 1.0).*sign(x)
end
