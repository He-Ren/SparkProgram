import math
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
import tensorcircuit as tc
import tensorflow as tf

K = tc.set_backend("tensorflow")

n = int(input())

I = tc.gates.i().tensor
X = tc.gates.x().tensor
Y = tc.gates.y().tensor
Z = tc.gates.z().tensor

def getmat(sp: dict[int : tf.Tensor]):
    a = [I] * n
    for i in sp:
        a[i] = sp[i]
    
    res = K.ones(shape = (1,1))
    for t in a:
        res = K.kron(res, t)
    return res

H = K.zeros(shape = (2**n, 2**n))
for i in range(n):
    H += getmat({i: Z})
for i in range(n-1):
    H += getmat({i: X}) @ getmat({i+1: X})

c = tc.Circuit(n)
res = c.expectation([H, range(n)])
print(res)