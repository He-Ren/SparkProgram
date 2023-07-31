import math
import numpy as np
from scipy import linalg as la
import tensorcircuit as tc
import tensorflow as tf
from matplotlib import pyplot as plt

def derivative2(f, x):
    delta = np.pi / 2
    tau = 2
    return (f(x + delta) - f(x - delta)) / tau

a = 11
b = 45
c = 14

def f(x):
    return a * np.sin(x + b) + c

def df(x):
    return a * np.cos(x + b)

vl = -np.pi
vr = np.pi

for x in np.linspace(vl, vr, 1024, endpoint = True):
    res1 = derivative2(f, x)
    res2 = df(x)
    assert math.isclose(res1, res2, abs_tol = 1e-9)