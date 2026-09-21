import numpy as np
import matplotlib.pyplot as plt

F32 = np.float32

def midpoint(f, a, b, N):
    a, b = F32(a), F32(b)
    h = (b - a) / F32(N)
    s = F32(0)
    for i in range(N):
        s = s + f(a + (F32(i) + F32(0.5)) * h)
    return h * s

def trapezoid(f, a, b, N):
    a, b = F32(a), F32(b)
    h = (b - a) / F32(N)
    s = (f(a) + f(b)) / F32(2)
    for i in range(1, N):
        s = s + f(a + F32(i) * h)
    return h * s

def simpson(f, a, b, N):
    a, b = F32(a), F32(b)
    h = (b - a) / F32(N)
    s = f(a) + f(b)
    for i in range(1, N):
        if i % 2 == 1:
            s = s + F32(4) * f(a + F32(i) * h)
        else:
            s = s + F32(2) * f(a + F32(i) * h)
    return h * s / F32(3)

def expminus(t):
    return np.exp(-t)

def actualint(a, b):
    return F32(np.exp(-a) - np.exp(-b))

def error(f, a, b, exact, Nmax):
    Ns = []
    err_m = []
    err_t = []
    err_s = []
    N = 2
    while N <= Nmax:
        im = midpoint(f, a, b, N)
        it = trapezoid(f, a, b, N)
        isim = simpson(f, a, b, N)
        Ns.append(N)
        err_m.append(abs((im - exact) / exact))
        err_t.append(abs((it - exact) / exact))
        err_s.append(abs((isim - exact) / exact))
        N = 2*N
    Ns = np.array(Ns)
    err_m = np.array(err_m)
    err_t = np.array(err_t)
    err_s = np.array(err_s)
    return Ns, err_m, err_t, err_s

exact = actualint(0.0, 1.0)
Ns, err_m, err_t, err_s = error(expminus, 0.0, 1.0, exact, 2**20)

plt.figure(figsize=(7, 5.5))
plt.loglog(Ns, err_m, label="midpoint")
plt.loglog(Ns, err_t, label="trapezoid")
plt.loglog(Ns, err_s, label="Simpson")

Nref = np.logspace(0.5, 2.5, 20)
plt.loglog(Nref, 1e-2 * (Nref / Nref[0]) ** -2.0, "--", color="0.6", lw=0.8)
plt.loglog(Nref, 1e-4 * (Nref / Nref[0]) ** -4.0, "--", color="0.6", lw=0.8)
Nref = np.logspace(3, 5.5, 20)
plt.loglog(Nref, 1e-9 * np.sqrt(Nref), ":", color="crimson", lw=0.8)

plt.xlabel("number of bins N")
plt.ylabel("relative error")
plt.title("$\\int_0^1 e^{-t}dt$, single precision")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("integration_error.png", dpi=150)
plt.show()