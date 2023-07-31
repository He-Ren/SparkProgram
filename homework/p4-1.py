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

print(c.draw())
print(c.expectation_ps(z = [0, 1]))
print(c.expectation([tc.gates.z(), [0]], [tc.gates.z(), [1]]))
print(c.expectation([zz, [0, 1]]))