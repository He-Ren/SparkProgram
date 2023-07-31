import math
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
import tensorcircuit as tc
import tensorflow as tf

n = int(input())

I = tc.gates._i_matrix
X = tc.gates._x_matrix
Y = tc.gates._y_matrix
Z = tc.gates._z_matrix

def getmat(sp: dict[int : np.array]):
    a = [I] * n
    for i in sp:
        a[i] = sp[i]
    
    res = np.array([1])
    for t in a:
        res = np.kron(res, t)
    return res

H = np.zeros(shape = (2**n, 2**n), dtype = complex)
for i in range(n):
    H += getmat({i: Z})
for i in range(n-1):
    H += getmat({i: X}) @ getmat({i+1: X})

print(f"H:\n{H}")

v = np.zeros(shape = (2**n), dtype = complex)
v[0] = 1

res = v.T.conj() @ H @ v
print(f"res = {res}")