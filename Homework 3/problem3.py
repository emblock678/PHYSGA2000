import sys
import numpy as np
import matplotlib.pyplot as plt
 
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = "blur.txt"
 
sigma = 25.0
eps = 1e-3

blur = np.loadtxt(filename)
K, L = blur.shape
print("image is %d x %d" % (K, L))
 
plt.figure(figsize=(6, 6))
plt.imshow(blur, cmap="gray", origin="upper")
plt.title("Blurred photo")
plt.axis("off")
plt.tight_layout()

def psf(K, L, sigma):
    y, x = np.meshgrid(np.arange(L), np.arange(K))
    x = np.where(x > K / 2, x - K, x)
    y = np.where(y > L / 2, y - L, y)
    return np.exp(-(x**2 + y**2) / (2 * sigma**2))

f = psf(K, L, sigma)
f = f / np.sum(f)
 
plt.figure(figsize=(6, 6))
plt.imshow(f, cmap="gray", origin="upper")
plt.title(r"Point spread function, $\sigma = %g$" % sigma)
plt.axis("off")
plt.tight_layout()

bk = np.fft.rfft2(blur)
fk = np.fft.rfft2(f)

ak = bk.copy()
big = np.abs(fk) > eps
ak[big] = bk[big] / fk[big]
sharp = np.fft.irfft2(ak, s=(K, L))

nskip = np.sum(np.abs(fk) <= eps)
print("coefficients left undivided: %d of %d (%.2f%%)"
      % (nskip, fk.size, 100 * nskip / fk.size))

plt.figure(figsize=(6, 6))
plt.imshow(sharp, cmap="gray", origin="upper")
plt.title("Deconvolved photo")
plt.axis("off")
plt.tight_layout()

fig, ax = plt.subplots(1, 2, figsize=(11, 5.6))
ax[0].imshow(blur, cmap="gray", origin="upper")
ax[0].set_title("blurred")
ax[0].axis("off")
ax[1].imshow(sharp, cmap="gray", origin="upper")
ax[1].set_title("deconvolved")
ax[1].axis("off")
plt.tight_layout()

kaxis = np.arange(L) - L // 2
mag = np.abs(np.fft.fftshift(np.fft.fft2(f)))
plt.figure(figsize=(7, 5))
plt.semilogy(kaxis, mag[K // 2, :], lw=1)
plt.axhline(eps, color="crimson", ls="--", label=r"$\epsilon = 10^{-3}$")
plt.xlabel(r"$k$ along one row")
plt.ylabel(r"$|\tilde{f}_{kl}|$")
plt.title("Transform of the point spread function (central row)")
plt.grid(alpha=0.3, which="both")
plt.legend()
plt.tight_layout()

plt.show()
