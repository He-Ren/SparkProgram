import numpy as np
import matplotlib.pylab as plt
from scipy import linalg as la
from numpy import pi

X = np.array([[0, 1],
              [1, 0]], dtype = complex)
Y = np.array([[0, -1j],
              [1j, 0]], dtype = complex)
Z = np.array([[1, 0],
              [0, -1]], dtype = complex)

def calc_E(Pid, Qid, theta):
    P = [X, Y, Z][Pid]
    Q = [X, Y, Z][Qid]
    v0 = np.array([1, 0], dtype = complex)
    v = la.expm(1j * theta / 2 * P) @ v0
    return v.conj().T @ Q @ v

plt.figure(figsize = (11, 11), dpi = 80)

for i in range(0, 3):
    for j in range(0, 3):
        
        plt.subplot(3, 3, i * 3 + j + 1)
        
        plt.xlim(-pi, pi)
        plt.xticks([-pi, 0, pi], ["$-\pi$", "$0$", "$+\pi$"])
        
        plt.ylim(-1.1, 1.1)
        plt.yticks([-1, 0, 1])
        
        Xs = np.linspace(-pi, pi, 256, endpoint = True)
        Ys = []
        for x in Xs:
            t = calc_E(i, j, x)
            np.testing.assert_allclose(t.imag, 0, atol = 1e-7)
            Ys.append(t.real)
        
        plt.title("$\\hat P={}, \\hat Q={}$".format("XYZ"[i], "XYZ"[j]));
        
        plt.plot(Xs, Ys, linewidth=2)

plt.show()