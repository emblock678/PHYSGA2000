import numpy as np
import matplotlib.pyplot as plt

tol = 10 ** -6

def f(x,c):
    return 1 - np.exp(- c * x)
def df(x,c):
    return c * np.exp(- c * x)

def relax(c):
    iterations = 1
    def error(x1, x2):
        return np.abs((x1 - x2) / (1 - 1 / df(x1,c)))

    x1 = f(0.5,c)
    x2 = 0.5
    while error(x1, x2) > tol:
        x1, x2 = f(x1,c), x1
        iterations += 1
    print('Iteration Count = ', iterations)
    return x1

def overrelax(c, omega):
    iterations = 1
    def error(x1, x2):
        return np.abs((x1 - x2) / (1 - 1 / ((1 + omega) * df(x1,c) - omega)))

    x1 = (1 + omega) * f(0.5,c) - omega * 0.5
    x2 = 0.5
    while error(x1, x2) > tol:
        x1, x2 = (1 + omega) * f(x1,c) - omega * x1, x1
        iterations += 1
    return x1, iterations

print(relax(2))

c = 2
ws = []
ns = []
omega = 0.0
while omega <= 1.5:
    x, n = overrelax(c, omega)
    ws.append(omega)
    ns.append(n)
    omega = omega + 0.01

ws = np.array(ws)
ns = np.array(ns)

i = ns.argmin()
print("fewest iterations: %d  at omega = %.2f" % (ns[i], ws[i]))

plt.figure(figsize=(7, 5))
plt.plot(ws, ns, ".-", lw=1, ms=4, label="overrelaxation")
plt.axhline(15, color="0.5", ls=":", label="simple relaxation (15)")
plt.xlabel(r"$\omega$")
plt.ylabel("Iterations")
plt.title("Overrelaxation for c = 2")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()