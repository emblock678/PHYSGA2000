import numpy as np
import matplotlib.pyplot as plt

F32 = np.float32

def forward(f, x, h):
    x, h = F32(x), F32(h)
    xp = x + h
    return (f(xp) - f(x)) / (xp - x)

def central(f, x, h):
    x, h = F32(x), F32(h)
    half = h / F32(2)
    xp = x + half
    xm = x - half
    return (f(xp) - f(xm)) / (xp - xm)

def extrapolated(f, x, h):
    h = F32(h)
    d_half = central(f, x, h / F32(2))
    d_full = central(f, x, h)
    return (F32(4) * d_half - d_full) / F32(3)

def actualex(a):
    actex = np.exp(a)
    return F32(actex)

def actualnsin(a):
    actnsin = -1*np.sin(a)
    return F32(actnsin)

def error(f,x,exact):
    h = F32(1)
    x=F32(x)
    hs = []
    err_f = []
    err_c = []
    err_e = []
    while h > 1e-7:
        df = forward(f,x,h)
        dc = central(f,x,h)
        de = extrapolated(f,x,h)
        hs.append(h)
        err_f.append(abs((df - exact) / exact))
        err_c.append(abs((dc - exact) / exact))
        err_e.append(abs((de - exact) / exact))
        h = h / F32(1.1)
    hs = np.array(hs)
    err_f = np.array(err_f)
    err_c = np.array(err_c)
    err_e = np.array(err_e)
    return hs, err_f, err_c, err_e

h1, f1, c1, e1 = error(np.cos, 0.1,  actualnsin(0.1))
h2, f2, c2, e2 = error(np.cos, 10.0, actualnsin(10.0))
h3, f3, c3, e3 = error(np.exp, 0.1,  actualex(0.1))
h4, f4, c4, e4 = error(np.exp, 10.0, actualex(10.0))

def panel(n, hs, err_f, err_c, err_e, title):
    plt.subplot(2, 2, n)
    plt.loglog(hs, err_f, label="forward")
    plt.loglog(hs, err_c, label="central")
    plt.loglog(hs, err_e, label="extrapolated")
    href = np.logspace(-3, -0.5, 20)
    plt.loglog(href, 1e-2*(href/href[-1])**1, "--", color="0.6", lw=0.8)
    plt.loglog(href, 1e-2*(href/href[-1])**2, "--", color="0.6", lw=0.8)
    plt.loglog(href, 1e-2*(href/href[-1])**4, "--", color="0.6", lw=0.8)
    href = np.logspace(-7, -4, 20)
    plt.loglog(href, 1e-8/href, ":", color="crimson", lw=0.8)
    plt.title(title)
    plt.xlabel("step size h")
    plt.ylabel("relative error")
    plt.ylim(1e-9, 1e2)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8, loc="lower left")

plt.figure(figsize=(11, 8))
panel(1, h1, f1, c1, e1, "cos(x) at x = 0.1")
panel(2, h2, f2, c2, e2, "cos(x) at x = 10")
panel(3, h3, f3, c3, e3, "exp(x) at x = 0.1")
panel(4, h4, f4, c4, e4, "exp(x) at x = 10")
plt.tight_layout()
plt.show()