import numpy as np
from scipy import linalg as la
import tensorcircuit as tc
import tensorflow as tf
from matplotlib import pyplot as plt

K = tc.set_backend("tensorflow")

def calc_np(Pid, Qid, theta):
    
    X = np.array([[0, 1],
              [1, 0]], dtype = complex)
    Y = np.array([[0, -1j],
                [1j, 0]], dtype = complex)
    Z = np.array([[1, 0],
                [0, -1]], dtype = complex)
    
    P = [X, Y, Z][Pid]
    Q = [X, Y, Z][Qid]
    v0 = np.array([1, 0], dtype = complex)
    v = la.expm(1j * theta / 2 * P) @ v0
    return (v.conj().T @ Q @ v).real

def calc_tc(Pi, Qi, theta):
    P = tc.gates.pauli_gates[Pi + 1].tensor
    Q = tc.gates.pauli_gates[Qi + 1].tensor
    
    c = tc.Circuit(1)
    c.exp1(0, theta = -theta / 2, unitary = P)
    
    return np.array(c.expectation([Q, [0]])).real

plt.figure(figsize = (12, 10), dpi = 80)

pi = np.pi

for i in range(0, 3):
    for j in range(0, 3):
        
        plt.subplot(3, 3, i * 3 + j + 1)
        
        plt.xlim(-pi, pi)
        plt.xticks([-pi, 0, pi], ["$-\pi$", "$0$", "$+\pi$"])
        
        plt.ylim(-1.1, 1.1)
        plt.yticks([-1, 0, 1])
        
        Xs = np.linspace(-pi, pi, 256, endpoint = True)
        Ys_np = []
        Ys_tc = []
        for x in Xs:
            Ys_np.append( calc_np(i, j, x) )
            Ys_tc.append( calc_tc(i, j, x) )
        
        np.testing.assert_allclose(Ys_np, Ys_tc, atol = 1e-6) 
        
        plt.title("$\\hat P={}, \\hat Q={}$".format("XYZ"[i], "XYZ"[j]));
        
        plt.plot(Xs, Ys_tc, linewidth=2)

plt.show()