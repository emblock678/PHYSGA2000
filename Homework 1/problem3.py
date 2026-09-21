import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

filename = "lcdm_z0.matter_pk"
kmax = 5.0
Nbins = 20000
damp = 1.0

def loadpk(name):
    data = np.loadtxt(name)
    k = data[:, 0]
    P = data[:, 1]
    return k, P

def makespline(k, P):
    return CubicSpline(np.log(k), np.log(P))

def pk(spline, k):
    return np.exp(spline(np.log(k)))

def integrand(spline, k, r, d):
    kr = k * r
    return k**3 * pk(spline, k) * np.sin(kr) / kr * np.exp(-(k * d)**2)

def simpson(y, h):
    s = y[0] + y[-1]
    s = s + 4 * np.sum(y[1:-1:2])
    s = s + 2 * np.sum(y[2:-1:2])
    return h * s / 3.0

def xi(spline, r, lnkmin, lnkmax, N, d):
    h = (lnkmax - lnkmin) / N
    lnk = lnkmin + h * np.arange(N + 1)
    y = integrand(spline, np.exp(lnk), r, d)
    return simpson(y, h) / (2.0 * np.pi**2)

def peak(r, y):
    best = -1
    for j in range(1, len(y) - 1):
        if y[j] > y[j-1] and y[j] > y[j+1]:
            if best < 0 or y[j] > y[best]:
                best = j
    i = best
    y0, y1, y2 = y[i-1], y[i], y[i+1]
    dr = r[i] - r[i-1]
    return r[i] + 0.5 * dr * (y0 - y2) / (y0 - 2*y1 + y2)

k, P = loadpk(filename)
spline = makespline(k, P)

lnkmin = np.log(k.min())
print(lnkmin)
lnkmax = np.log(min(kmax, k.max()))

rs = []
xis = []
r = 50.0
while r <= 120.0:
    rs.append(r)
    xis.append(xi(spline, r, lnkmin, lnkmax, Nbins, damp))
    r = r + 0.25

rs = np.array(rs)
xis = np.array(xis)
r2xi = rs**2 * xis

rpeak = peak(rs, r2xi)
print("BAO peak at r = %.2f Mpc/h" % rpeak)

plt.figure(figsize=(7.5, 5.5))
plt.plot(rs, r2xi, lw=1.8, label=r"$r^2\xi(r)$")
#plt.plot(rs, r2xi, ".", color="green", label="computed points")
plt.axvline(rpeak, color="crimson", ls="--", lw=1.2,
            label="BAO peak, r = %.1f Mpc/h" % rpeak)
plt.plot(rpeak, np.interp(rpeak, rs, r2xi), "o", color="crimson")
plt.xlabel("r  [Mpc/h]")
plt.ylabel(r"$r^2\,\xi(r)$   [(Mpc/h)$^2$]")
plt.title("Correlation function and the BAO peak")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("bao_peak.png", dpi=150)
plt.show()