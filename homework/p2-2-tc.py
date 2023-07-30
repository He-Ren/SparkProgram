import numpy as np

import tensorcircuit as tc
import tensorflow as tf

K = tc.set_backend("tensorflow")

c = tc.Circuit(1, inputs = [1,0])

P = tc.gates._y_matrix
theta = 1

print(f"{P = }")

c.exp1(0, theta = -theta, unitary = P)

print(c.to_qir())
print(c.state())
print(c.draw())