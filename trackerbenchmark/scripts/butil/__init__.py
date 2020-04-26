import math
import sys


def d_to_f(x):
    return [round(float(o), 4) for o in x]


def matlab_double_to_py_float(double):
    return list(map(d_to_f, double))


def ssd(x, y):
    if len(x) != len(y):
        sys.exit("cannot calculate ssd")
    s = 0
    for i in range(len(x)):
        s += (x[i] - y[i]) ** 2
    return math.sqrt(s)
