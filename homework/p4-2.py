import math
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
import tensorcircuit as tc
import tensorflow as tf

K = tc.set_backend("tensorflow")

c = tc.Circuit(2)
c.h(0)
c.cx(0, 1)

zz = K.kron(tc.gates.z().tensor, tc.gates.z().tensor)

sum = 0
T = 100
for _ in range(T):
    cur = c.perfect_sampling() [0]
    idx = 0
    for i in cur:
        idx = idx * 2 + int(i)
    sum += zz[idx][idx]
sum /= T

print(sum)