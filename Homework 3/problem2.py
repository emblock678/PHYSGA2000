import sys
import numpy as np
import matplotlib.pyplot as plt
 
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = "dow.txt"

dow = np.loadtxt(filename)
N = len(dow)
day = np.arange(N)
 
print("N = ", N, " business days")

c = np.fft.rfft(dow)
print("rfft returns ", len(c), " coefficients (N/2 + 1 = ", N // 2 + 1, ")")

def smooth(c, frac):
    cc = c.copy()
    keep = int(frac * len(cc))
    cc[keep:] = 0
    return np.fft.irfft(cc, n=N), keep

y10, keep10 = smooth(c, 0.10)
y02, keep02 = smooth(c, 0.02)

print("10%% filter keeps %d coefficients, highest frequency = %.1f days"
      % (keep10, N / keep10))
print(" 2%% filter keeps %d coefficients, highest frequency = %.1f days"
      % (keep02, N / keep02))

plt.figure(figsize=(10, 5.5))
plt.plot(day, dow, lw=0.8, color="0.6", label="original data")
plt.plot(day, y10, lw=1.4, color="C0", label="first 10% of coefficients")
plt.plot(day, y02, lw=1.6, color="crimson", label="first 2% of coefficients")
plt.xlabel("Time since Late 2006 (business days)")
plt.ylabel("Dow Jones Industrial Average (points)")
plt.title("Fourier Filtering of the Dow Jones Index")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()