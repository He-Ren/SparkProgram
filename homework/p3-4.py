import math
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
import tensorcircuit as tc
import tensorflow as tf

K = tc.set_backend("tensorflow")

def getf(Pi, Qi):
    P = tc.gates.pauli_gates[Pi + 1].tensor
    Q = tc.gates.pauli_gates[Qi + 1].tensor
    
    def f(param):
        
        theta = param[0]
        
        c = tc.Circuit(1)
        c.exp1(0, theta = - theta / 2, unitary = P)
        
        return K.real( c.expectation([Q, [0]]) )
    
    return f

plt.figure(figsize = (10, 10))

for Pi in range(3):
    for Qi in range(3):
        
        f = getf(Pi, Qi)
        vgf = K.value_and_grad(f)

        param = K.implicit_randn([1,1])
        history_x = []
        history_y = []
        opt = K.optimizer(tf.keras.optimizers.Adam(0.01))

        T = 500
        for _ in range(T):
            v,g = vgf(param)
            history_x.append(param[0])
            history_y.append(v)
            param = opt.update(g, param)
        
        plt.subplot(3, 3, Pi * 3 + Qi + 1)
        plt.title("$\\hat P={}, \\hat Q={}$".format("XYZ"[Pi], "XYZ"[Qi]));
        
        plt.plot(range(T), history_y)

plt.show()