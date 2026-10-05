import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = "smf_cosmos.dat"
tol = 10 ** -6
hd = 10 ** -5

def grad(F, p):
    g = np.zeros(len(p))
    i = 0
    while i < len(p):
        dp = np.zeros(len(p))
        dp[i] = hd * max(abs(p[i]), 1.0)
        g[i] = (F(p + dp) - F(p - dp)) / (2 * dp[i])
        i = i + 1
    return g

def descent(F, p, step, maxit):
    p = np.array(p, dtype=float)
    ps = [p.copy()]
    fs = [F(p)]
    it = 0
    while it < maxit:
        g = grad(F, p)
        gmag = np.sqrt(np.sum(g**2))
        if gmag < tol:
            break
        pnew = p - step * g / gmag
        fnew = F(pnew)
        if fnew < fs[-1]:
            p = pnew
            ps.append(p.copy())
            fs.append(fnew)
            step = step * 1.2
            it = it + 1
        else:
            step = step / 2
            if step < 10 ** -14:
                break
    return np.array(ps), np.array(fs)

def test(p):
    return (p[0] - 2)**2 + (p[1] - 2)**2

ps, fs = descent(test, [-1.0, 3.5], 0.5, 1000)
print("test function: minimum at", ps[-1], " after", len(fs) - 1, "steps")

data = np.loadtxt(filename)
logM = data[:, 0]
n = data[:, 1]
err = data[:, 2]

def schechter(logM, p):
    # p = [log10 phi*, log10 M*, alpha]
    r = 10 ** (logM - p[1])
    return np.log(10) * 10**p[0] * r**(p[2] + 1) * np.exp(-r)

def chi2(p):
    return np.sum(((schechter(logM, p) - n) / err)**2)

starts = [[-2.0, 10.0, -1.0],
          [-4.0, 11.5, -1.8],
          [-3.0, 10.5, -0.5],
          [-1.5, 11.9, -1.5]]

runs = []
i = 0
while i < len(starts):
    ps, fs = descent(chi2, starts[i], 0.1, 20000)
    runs.append((ps, fs))
    print("start %-22s -> log phi* = %7.4f  log M* = %7.4f  alpha = %7.4f"
          "   chi2 = %8.4f  (%d steps)"
          % (str(starts[i]), ps[-1][0], ps[-1][1], ps[-1][2], fs[-1], len(fs) - 1))
    i = i + 1

best = runs[0][0][-1]
print("\nbest fit: phi* = %.4e  M* = %.4e  alpha = %.4f"
      % (10**best[0], 10**best[1], best[2]))
print("chi2 = %.3f for %d - 3 = %d dof  ->  chi2/dof = %.3f"
      % (chi2(best), len(n), len(n) - 3, chi2(best) / (len(n) - 3)))

#test function
plt.figure(figsize=(6.5, 5.5))
ps, fs = descent(test, [-1.0, 3.5], 0.5, 1000)
gx, gy = np.meshgrid(np.linspace(-1.5, 4, 200), np.linspace(-1, 4, 200))
plt.contour(gx, gy, (gx - 2)**2 + (gy - 2)**2, levels=20, colors="0.8")
plt.plot(ps[:, 0], ps[:, 1], "o-", ms=4, lw=1, label="descent path")
plt.plot(2, 2, ".", ms=15, color="crimson", label="true minimum (2, 2)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Test: $f(x,y)=(x-2)^2+(y-2)^2$")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.figure(figsize=(7, 5))
i = 0
while i < len(runs):
    plt.semilogy(runs[i][1], lw=1.2, label="start %s" % str(starts[i]))
    i = i + 1
plt.xlabel("step i")
plt.ylabel(r"$\chi^2$")
plt.title(r"$\chi^2$ vs step, from four starting points")
plt.grid(alpha=0.3, which="both")
plt.legend(fontsize=8)
plt.tight_layout()

plt.figure(figsize=(7, 5.5))
mfine = np.linspace(logM.min() - 0.2, logM.max() + 0.2, 400)
plt.errorbar(10**logM, n, yerr=err, fmt="o", ms=4, capsize=3, label="COSMOS data")
plt.plot(10**mfine, schechter(mfine, best), lw=1.5, label="best-fit Schechter")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$M_{gal}$  [$M_\odot$]")
plt.ylabel(r"$n(M_{gal})$  [dex$^{-1}$ (Mpc/h)$^{-3}$]")
plt.title("Stellar mass function")
plt.grid(alpha=0.3, which="both")
plt.legend()
plt.tight_layout()
plt.show()