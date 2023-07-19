import numpy as np
import math
from numpy import matlib, sin, cos, array, matmul, pi
from matplotlib import pyplot as plt
from scipy import linalg as la

def exp_iP(P):
    e, U = la.eigh(P)
    U = U.conj().T
    
    np.testing.assert_allclose(U.conj().T @ np.diag(e) @ U, P, atol = 1e-8)
    np.testing.assert_allclose(U.conj().T @ U, matlib.identity(2), atol = 1e-8)
    
    return U.conj().T @ np.diag(np.exp(1j * e)) @ U

def check(P):
    
    np.testing.assert_allclose(P @ P, matlib.identity(2), 1e-8)
    
    A = la.expm(1j * P)
    B = exp_iP(P)
    
    np.testing.assert_allclose(A, B, atol = 1e-8)
    
    for theta in np.linspace(-pi, pi, 256):
        A = exp_iP(P * theta)
        B = cos(theta) * matlib.identity(2) + sin(theta) * 1j * P
        np.testing.assert_allclose(A, B, atol = 1e-8)

check(array([[0, 1], [1, 0]], dtype = complex))
check(array([[0, -1j], [1j, 0]], dtype = complex))
check(array([[1, 0], [0, -1]], dtype = complex))
