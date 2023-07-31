import math
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
import tensorcircuit as tc
import tensorflow as tf

K = tc.set_backend("tensorflow")

def derivative1(f, x):
    delta = 0.0001
    return (f(x + delta) - f(x)) / delta

def derivative2(f, x):
    delta = np.pi / 2
    tau = 2
    return (f(x + delta) - f(x - delta)) / tau

def getf(Pi, Qi):
    P = tc.gates.pauli_gates[Pi + 1].tensor
    Q = tc.gates.pauli_gates[Qi + 1].tensor
    
    def f(param):
        
        theta = param[0]
        
        c = tc.Circuit(1)
        c.exp1(0, theta = - theta / 2, unitary = P)
        
        return K.real( c.expectation([Q, [0]]) )
    
    return f


for Pi in range(3):
    for Qi in range(3):
        
        f = getf(Pi, Qi)
        
        vl = 0
        vr = np.pi * 2
        
        for tmpx in np.linspace(vl, vr, 256, endpoint = True):
            
            x = K.convert_to_tensor([tmpx])
            res1 = derivative1(f, x)
            res2 = derivative2(f, x)
            res3 = K.grad(f) (x)
            
            ok = 1
            ok &= math.isclose(res1, res3, abs_tol = 0.01)
            ok &= math.isclose(res2, res3, abs_tol = 0.01)
            
            if ok == False:
                print(f"{res1 = }, {res2 = }, {res3 = }")
            
            assert ok 

plt.show()