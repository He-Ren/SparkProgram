import numpy as np

import tensorcircuit as tc
import tensorflow as tf

K = tc.set_backend("tensorflow")

def calc_R(theta):
    return K.convert_to_tensor( np.array([
        [tf.cos(theta), tf.sin(theta)],
        [-tf.sin(theta), tf.cos(theta)]
    ], dtype = complex))

v = K.convert_to_tensor( np.array( [[0.], [1.]], dtype = complex ) )
theta = 1 + 0j

res = calc_R(theta) @ v

print(f"{v = }")
print(f"{res = }")