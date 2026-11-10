import numpy as np
import matplotlib.pyplot as plt

a = 1.0
b = 3.0
delta = 1e-10          # target accuracy per unit time
nmax = 8               # max modified-midpoint steps before splitting H

def f(r, t):
    x = r[0]
    y = r[1]
    fx = 1 - (b + 1) * x + a * x**2 * y
    fy = b * x - a * x**2 * y
    return np.array([fx, fy], float)

def modmid(r, t, H, n):
    # modified midpoint rule: n steps of size h across the interval H
    h = H / n
    r1 = r + 0.5 * h * f(r, t)
    r2 = r + h * f(r1, t + 0.5 * h)
    i = 1
    while i < n:
        r1 = r1 + h * f(r2, t + i * h)
        r2 = r2 + h * f(r1, t + (i + 0.5) * h)
        i = i + 1
    return 0.5 * (r1 + r2 + 0.5 * h * f(r2, t + H))

def bsstep(r, t, H, tpoints, rpoints):
    # one adaptive Bulirsch-Stoer step across [t, t+H].
    # try n = 1,2,...,nmax; if none converges, halve H and recurse
    R1 = None
    n = 1
    while n <= nmax:
        R2 = np.empty([n, 2], float)
        # a large H can make the midpoint rule overflow; that trial is simply
        # rejected, so the warnings it raises are suppressed
        with np.errstate(over="ignore", invalid="ignore"):
            R2[0] = modmid(r, t, H, n)
        if not np.all(np.isfinite(R2[0])):
            break
        if n > 1:
            m = 1
            while m < n:
                eps = (R2[m-1] - R1[m-1]) / ((n / (n - 1))**(2 * m) - 1)
                R2[m] = R2[m-1] + eps
                m = m + 1
            # require the target accuracy in BOTH x and y
            if max(abs(eps[0]), abs(eps[1])) < H * delta:
                tpoints.append(t + H)
                rpoints.append(R2[n-1])
                return R2[n-1]
        R1 = R2
        n = n + 1
    # no convergence within nmax steps: split the interval in half
    rmid = bsstep(r, t, H / 2, tpoints, rpoints)
    return bsstep(rmid, t + H / 2, H / 2, tpoints, rpoints)

t0 = 0.0
tf = 20.0
H = 20.0
r = np.array([0.0, 0.0], float)

tpoints = [t0]
rpoints = [r.copy()]

bsstep(r, t0, H, tpoints, rpoints)

tpoints = np.array(tpoints)
rpoints = np.array(rpoints)
x = rpoints[:, 0]
y = rpoints[:, 1]

print("intervals accepted  = ", len(tpoints) - 1)
print("smallest interval   = %.3e" % np.min(np.diff(tpoints)))
print("largest interval    = %.3e" % np.max(np.diff(tpoints)))
print("x, y at t = 20      = %.10f  %.10f" % (x[-1], y[-1]))

plt.figure(figsize=(10, 5.5))
plt.plot(tpoints, x, "-", lw=1.2, color="C0", label="x(t)")
plt.plot(tpoints, x, ".", ms=5, color="C0")
plt.plot(tpoints, y, "-", lw=1.2, color="crimson", label="y(t)")
plt.plot(tpoints, y, ".", ms=5, color="crimson")
plt.xlabel("t")
plt.ylabel("concentration")
plt.title("Brusselator, a = %g, b = %g (dots mark time-interval boundaries)"
          % (a, b))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("p1_brusselator.png", dpi=150)

# interval size against time, to show the adaptation directly
plt.figure(figsize=(10, 4))
plt.semilogy(tpoints[1:], np.diff(tpoints), ".-", lw=1, ms=5)
plt.xlabel("t")
plt.ylabel("interval size H")
plt.title("Size of the accepted time intervals")
plt.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("p1_stepsize.png", dpi=150)
plt.show()