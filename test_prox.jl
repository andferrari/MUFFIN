using Images
using FITSIO
using PyPlot

f = FITS("/Users/ferrari/.julia/v0.4/MegaMage/data/monarch.fits")
x0 = read(f[1])
close(f)

function gauss_filt{T<:Float64}(hsize::Int, sigma::T)
           linx = linspace(-(hsize-1)/2,(hsize-1)/2,hsize)
           gridx = [x for x in linx, j in 1:hsize]
           psf = exp(-(gridx.^2 + transpose(gridx).^2)/(2*sigma*sigma))
           psf = psf / sum(psf)
       end

psf = gauss_filt(31,5.0)
psfadj = psf

y = imfilter(x0, psf, "circular") + randn(size(x0))

τ = 0.95/(0.5)
c = Float64[]

x = y
for k = 1:128
  ∇ =  imfilter(imfilter(x, psf, "circular") - y, psfadj, "circular")
  x = max(x - τ*∇, 0.0)
  push!(c,vecnorm(y - imfilter(x, psf, "circular")))
end


figure()
imshow(x,cmap=ColorMap("gray"))

figure()
imshow(x0,cmap=ColorMap("gray"))
title("original")

figure()
plot(c)

figure()
imshow(y,cmap=ColorMap("gray"))
title("noisy")
