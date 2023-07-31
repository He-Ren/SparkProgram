import math
import numpy as np
from scipy import linalg as la
import tensorcircuit as tc
import tensorflow as tf
from matplotlib import pyplot as plt

def derivative(f, x):
    delta = 1e-8
    return (f(x + delta) - f(x)) / delta

a = 11
b = 45
c = 14

def f(x):
    return a * x * x + b * x + c

def df(x):
    return 2 * a * x + b

vl = -1000
vr = 1000

for x in np.linspace(vl, vr, 1024, endpoint = True):
    res1 = derivative(f, x)
    res2 = df(x)
    assert math.isclose(res1, res2, rel_tol = 1e-3)