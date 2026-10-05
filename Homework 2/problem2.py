import numpy as np
import matplotlib.pyplot as plt

tol = 10 ** -6

h = 6.62607015e-34
c = 2.99792458e8
kB = 1.380649e-23
L = 502e-9

def f(x):
    return 5 * np.exp(-x) + x - 5

def binary(x1, x2):
    iterations = 0
    f1 = f(x1)
    while np.abs(x2 - x1) > tol:
        xm = (x1 + x2) / 2
        fm = f(xm)
        if f1 * fm > 0:
            x1, f1 = xm, fm
        else:
            x2 = xm
        iterations += 1
    print('Iteration Count = ', iterations)
    return (x1 + x2) / 2

x = binary(1.0, 10.0)
b = h * c / (kB * x)
 
print('x = ', x)
print('b = ', b, ' m K')
T = b / L
print('T_sun = ', T, ' K')