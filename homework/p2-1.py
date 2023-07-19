import numpy as np
import numpy.matlib
from numpy import sin, cos, array, matmul, pi
from matplotlib import pyplot as plt

def rotate_mat(theta):
    return array([ [cos(theta), -sin(theta)]
                  ,[sin(theta),  cos(theta)]])

def rotate(v, theta):
    return matmul(rotate_mat(theta), v)

def draw_seg(u,v,step=256):
    xs = np.linspace(u[0], v[0], step)
    ys = np.linspace(u[1], v[1], step)
    plt.plot(xs, ys)

def draw_vec(v,step=256):
    draw_seg(array([0,0]), v, step)

plt.xlim(-3, 3)
plt.ylim(-3, 3)

v = array([1, 0])
theta = 2 * pi / 10
res = rotate(v, theta)

print(v)
print(res)

draw_vec(v)
draw_vec(res)

plt.show()