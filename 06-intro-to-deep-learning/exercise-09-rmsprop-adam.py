import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time


def rosenbrock_function(x, y, a=1, b=100):


    return (a - x)**2 + b * (y - x**2)**2


def rosenbrock_gradient(x, y, a=1, b=100):

    df_dx = -2 * (a - x) - 4 * b * x * (y - x**2)
    df_dy = 2 * b * (y - x**2)
    return np.array([df_dx, df_dy])