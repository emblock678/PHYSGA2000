import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# constants
pc = 3.0856775814913673e16
Gsi = 6.67430e-11
csi = 2.99792458e8
Myr = 3.1556952e13

Lunit = 100 * pc
rs = 1e-7
Mbh = rs * Lunit * csi**2 / (2 * Gsi)
Tunit = sqrt(Lunit**3 / (Gsi * Mbh))

kgrav = 0.25
delta = 1e-10
safe = 0.9

print("M_BH    = %.4e kg = %.4e Msun" % (Mbh, Mbh / 1.98892e30))
print("1 time unit = %.4e s = %.4f Myr" % (Tunit, Tunit / Myr))
print("v_circ at r=1 = %.4f,  0.8 v_circ = %.4f" % (sqrt(kgrav), 0.8 * sqrt(kgrav)))

def deriv(x, y, vx, vy, A, B):
    rad = sqrt(x * x + y * y)
    v = sqrt(vx * vx + vy * vy)
    g = -kgrav / (rad * rad * rad)
    d = A / (v * v * v + B)
    return vx, vy, g * x - d * vx, g * y - d * vy

def rk4step(s, h, A, B):
    x, y, vx, vy = s
    a1, a2, a3, a4 = deriv(x, y, vx, vy, A, B)
    b1, b2, b3, b4 = deriv(x + 0.5*h*a1, y + 0.5*h*a2,
                           vx + 0.5*h*a3, vy + 0.5*h*a4, A, B)
    c1, c2, c3, c4 = deriv(x + 0.5*h*b1, y + 0.5*h*b2,
                           vx + 0.5*h*b3, vy + 0.5*h*b4, A, B)
    d1, d2, d3, d4 = deriv(x + h*c1, y + h*c2, vx + h*c3, vy + h*c4, A, B)
    s6 = h / 6
    return (x + s6 * (a1 + 2*b1 + 2*c1 + d1),
            y + s6 * (a2 + 2*b2 + 2*c2 + d2),
            vx + s6 * (a3 + 2*b3 + 2*c3 + d3),
            vy + s6 * (a4 + 2*b4 + 2*c4 + d4))

def integrate(A, B, v0, tmax, tol=delta, h=1e-5, rstop=0.0, keep=True):
    s = (1.0, 0.0, 0.0, v0)
    t = 0.0
    ts = [t]; xs = [s[0]]; ys = [s[1]]; vxs = [s[2]]; vys = [s[3]]
    nrej = 0
    while t < tmax:
        s1 = rk4step(rk4step(s, h, A, B), h, A, B)
        s2 = rk4step(s, 2 * h, A, B)
        ex = s1[0] - s2[0]
        ey = s1[1] - s2[1]
        err = sqrt(ex * ex + ey * ey) / 30
        rho = 4.0 if err == 0.0 else h * tol / err
        if rho >= 1.0:
            t = t + 2 * h
            s = s1
            if keep:
                ts.append(t); xs.append(s[0]); ys.append(s[1])
                vxs.append(s[2]); vys.append(s[3])
            if rstop > 0 and sqrt(s[0]**2 + s[1]**2) < rstop:
                break
            q = safe * rho**0.25
            h = h * (q if q < 2.0 else 2.0)
        else:
            nrej = nrej + 1
            q = safe * rho**0.25
            h = h * (q if q > 0.2 else 0.2)
    if not keep:
        return t
    return (np.array(ts), np.array(xs), np.array(ys),
            np.array(vxs), np.array(vys), nrej)

# No drag

v0a = sqrt(2 * kgrav * rs / (1 + rs))
E0 = 0.5 * v0a**2 - kgrav
L0 = v0a
aaxis = -kgrav / (2 * E0)
period = 2 * np.pi * sqrt(aaxis**3 / kgrav)

print("\npart (a)")
print("v0 = %.8e   a = %.6f   period = %.6f" % (v0a, aaxis, period))

ta, xa, ya, vxa, vya, nreja = integrate(0.0, 1.0, v0a, 10 * period)
rada = np.hypot(xa, ya)
Ea = 0.5 * (vxa**2 + vya**2) - kgrav / rada
La = xa * vya - ya * vxa

peri = []
i = 1
while i < len(rada) - 1:
    if rada[i] < rada[i-1] and rada[i] <= rada[i+1]:
        peri.append((ta[i], rada[i]))
    i = i + 1

print("steps = %d   rejected = %d" % (len(ta) - 1, nreja))
print("|dE/E| = %.3e   |dL/L| = %.3e" % (abs((Ea[-1]-E0)/E0), abs((La[-1]-L0)/L0)))
print("pericenter passages:")
j = 0
while j < len(peri):
    print("   orbit %2d   t = %8.4f   r_peri = %.8e" % (j+1, peri[j][0], peri[j][1]))
    j = j + 1

fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
ax[0].semilogy(ta, rada, lw=0.9)
ax[0].axhline(rs, color="crimson", ls="--", lw=1, label=r"$r_s = 10^{-7}$")
ax[0].set_ylabel("r")
ax[0].set_title(r"Frictionless orbit, $r_{peri}=r_s$, $\delta = %.0e$" % delta)
ax[0].grid(alpha=0.3, which="both")
ax[0].legend()
ax[1].plot(ta, abs((Ea - E0) / E0), lw=0.9, label="|dE/E|")
ax[1].plot(ta, abs((La - L0) / L0), lw=0.9, label="|dL/L|")
ax[1].set_yscale("log")
ax[1].set_xlabel("t")
ax[1].set_ylabel("relative drift")
ax[1].grid(alpha=0.3, which="both")
ax[1].legend()
plt.tight_layout()
plt.savefig("p2a_tolerance.png", dpi=150)

# A=B=1

vcirc = sqrt(kgrav)
v0 = 0.8 * vcirc

print("\npart (b)")
tb, xb, yb, vxb, vyb, nrejb = integrate(1.0, 1.0, v0, 1e4, rstop=rs)
radb = np.hypot(xb, yb)
print("steps = %d   rejected = %d" % (len(tb) - 1, nrejb))
print("reached r = %.3e at t = %.5f  (%.4f Myr)"
      % (radb[-1], tb[-1], tb[-1] * Tunit / Myr))

plt.figure(figsize=(9, 3.8))
plt.plot(xb, yb, lw=0.5)
plt.plot([0], [0], "+", ms=11, color="crimson", label="center of mass")
plt.plot([1], [0], "o", ms=6, color="C2", label="start")
plt.gca().set_aspect("equal")
plt.xlabel("x  [100 pc]")
plt.ylabel("y  [100 pc]")
plt.title("BH path with dynamical friction, A = B = 1")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("p2b_path.png", dpi=150)

plt.figure(figsize=(9, 5))
plt.semilogy(tb, radb, lw=0.9)
plt.axhline(rs, color="crimson", ls="--", lw=1, label=r"$r_s = 10^{-7}$")
plt.xlabel("t")
plt.ylabel("r  [100 pc]")
plt.title("Orbital radius against time, A = B = 1")
plt.grid(alpha=0.3, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("p2b_logr.png", dpi=150)

# B/A dependance

vals = [0.5, 1.0, 2.0, 5.0, 10.0]

print("\npart (c)")
print("%6s %6s %8s %14s %12s" % ("A", "B", "B/A", "t_merge", "Myr"))
sweepA = []; sweepB = []; sweepT = []
ia = 0
while ia < len(vals):
    ib = 0
    while ib < len(vals):
        A = vals[ia]; B = vals[ib]
        tm = integrate(A, B, v0, 1e5, rstop=rs, keep=False)
        sweepA.append(A); sweepB.append(B); sweepT.append(tm)
        print("%6.1f %6.1f %8.2f %14.5f %12.4f"
              % (A, B, B / A, tm, tm * Tunit / Myr))
        ib = ib + 1
    ia = ia + 1

sweepA = np.array(sweepA); sweepB = np.array(sweepB); sweepT = np.array(sweepT)
np.savetxt("sweep.dat", np.column_stack([sweepA, sweepB, sweepT]),
           header="A  B  t_merge", fmt="%.6e")

plt.figure(figsize=(8.5, 6))
ia = 0
while ia < len(vals):
    m = sweepA == vals[ia]
    o = np.argsort(sweepB[m] / sweepA[m])
    plt.loglog((sweepB[m] / sweepA[m])[o], (sweepT[m] * Tunit / Myr)[o],
               "o-", ms=5, lw=1.2, label="A = %g" % vals[ia])
    ia = ia + 1

q = np.logspace(np.log10(1.0), np.log10(25), 20)
plt.loglog(q, 12.8 * q * Tunit / Myr, "--", color="0.5", lw=1.2,
           label=r"underdamped: $t \propto \,B/A$")
q = np.logspace(np.log10(0.04), np.log10(0.6), 20)
plt.loglog(q, (1 / (3 * kgrav)) / q * Tunit / Myr, ":", color="0.3", lw=1.4,
           label=r"overdamped: $t \propto \,A/B$")
 
plt.xlabel("B / A")
plt.ylabel("time to reach $r_s$  [Myr]")
plt.title("Inspiral time against B/A, for each value of A")
plt.grid(alpha=0.3, which="both")
plt.legend(fontsize=9, loc="upper left", ncol=2)
plt.tight_layout()
plt.savefig("p2c_sweep.png", dpi=150)

# v0 dependence

print("\npart (d)")
fracs = [0.5, 0.65, 0.8, 0.95]
vlist = []; tlist = []
i = 0
while i < len(fracs):
    vv = fracs[i] * vcirc
    tm = integrate(1.0, 1.0, vv, 1e5, rstop=rs, keep=False)
    vlist.append(fracs[i]); tlist.append(tm)
    print("v0 = %.2f v_circ   t_merge = %10.5f   (%.4f Myr)"
          % (fracs[i], tm, tm * Tunit / Myr))
    i = i + 1

plt.figure(figsize=(7, 5))
plt.plot(vlist, np.array(tlist) * Tunit / Myr, "o-", ms=6)
plt.xlabel(r"$v_0 / v_{circ}$")
plt.ylabel("time to reach $r_s$  [Myr]")
plt.title("Inspiral time against initial velocity, A = B = 1")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("p2d_velocity.png", dpi=150)
plt.show()