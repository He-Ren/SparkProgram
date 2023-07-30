import numpy as np
import tensorcircuit as tc
import tensorflow as tf
from matplotlib import pyplot as plt

K = tc.set_backend("tensorflow")

c = tc.Circuit(1, inputs = [1,0])

print(c.state())

theta = 1

c.ry(0, theta = - theta * 2)

print(c.to_qir())
print(c.draw())
print(c.state())