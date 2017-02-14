using FITSIO
using PyPlot

f = FITS("/Users/ferrari/Desktop/Rita_Magellan/easy_muffin/data256/monarch.fits")
x0 = read(f[1])[1:256,1:256]
close(f)


imfilter(x,h) = real(ifftshift(ifft(fft(x).*conj(fft(h)))))

function gauss_filt{T<:Float64}(hsize::Int, sigma::T)
           linx = linspace(-(hsize-1)/2,(hsize-1)/2,hsize)
           gridx = [x for x in linx, j in 1:hsize]
           psf = exp(-(gridx.^2 + transpose(gridx).^2)/(2*sigma*sigma))
           psf = psf / sum(psf)
end

# Gauss psf
psf = gauss_filt(256,5.0)
β =1

β = maxabs(fft(psf))^2
psfadj = circshift(flipdim(flipdim(psf,2),1), (1, 1))
y = imfilter(x0, psf) + randn(size(x0))

τ = 0.99/(0.5β)
c = Float64[]

x = y
for k = 1:400
  println(k)
  ∇ =  imfilter(imfilter(x, psf) - y, psfadj)
  x = max(x - τ*∇, 0.0)
  push!(c,vecnorm(y - imfilter(x, psf)))
end


figure()
subplot(221)
imshow(x0,cmap=ColorMap("gray"))
title("original image")

subplot(222)
imshow(y,cmap=ColorMap("gray"))
title("noisy  image")

subplot(223)
imshow(x,cmap=ColorMap("gray"))
title("rconstructed")

subplot(224)
plot(c)
title("Cost")

suptitle("ISTA")
