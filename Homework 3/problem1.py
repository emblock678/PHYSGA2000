import sys
import numpy as np
import matplotlib.pyplot as plt
 
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = "sunspots.txt"

data = np.loadtxt(filename)
month = data[:, 0]
spots = data[:, 1]
N = len(spots)

plt.figure(figsize=(10, 4.5))
plt.plot(month, spots, lw=0.8)
plt.xlabel("Time since January 1749 (months)")
plt.ylabel("Sunspot Count")
plt.title("Sunspot Count vs. Time")
plt.grid(alpha=0.3)
plt.tight_layout()

c = np.fft.rfft(spots)
power = np.abs(c)**2
k = np.arange(len(power))
 
plt.figure(figsize=(7.5, 5))
plt.plot(k, power, lw=0.9)
plt.xlabel(r"$k$")
plt.ylabel(r"$|c_k|^2$")
plt.title("Power Spectrum")
plt.yscale("log")
plt.grid(alpha=0.3, which="both")
plt.tight_layout()

kpeak = 1 + power[1:].argmax()
period = N / kpeak
 
print("peak at k = ", kpeak)
print("period = N/k = %.2f months = %.2f years" % (period, period / 12))
 
plt.figure(figsize=(7.5, 5))
plt.plot(k[1:100], power[1:100], ".-", lw=1, ms=4)
plt.axvline(kpeak, color="crimson", ls="--",
            label="k = %d, period = %.0f months" % (kpeak, period))
plt.xlabel(r"$k$")
plt.ylabel(r"$|c_k|^2$")
plt.title("Power Spectrum at Low k")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()