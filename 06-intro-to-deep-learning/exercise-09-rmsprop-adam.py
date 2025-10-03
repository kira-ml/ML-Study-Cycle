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




class RMSProp:
    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.cache = None

    def update(self, parameters, gradient):

        if self.cache is None:
            self.cache = np.zeros_like(parameters)
            
            
            
        self.cache = self.beta * self.cache + (1 - self.beta) * (gradients ** 2)\
        

        parameters -= self.lr * gradients / (np.sqrt(self.cache) +  self.epsilon)

        return parameters



class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, parameters, gradients):
        if self.m is None:
            self.m = np.zeros_like(parameters)
            self.v = np.zeros_like(parameters)
        
        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients


        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)

        v_hat = self.v / (1 - self.beta2 ** self.t)

        parameterrs -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return parameters



    
    def test_optimizers_on_rosenbrock():


        start_point = np.array([-1.5, 2.0])
        target_point = np.array([1.0, 1.0])


        rmsprop = RMSProp(learning_rate=0.01)
        adam = Adam(learning_rate=0.1)


        points_rmsprop = [start_point.copy()]
        points_adam = [start_point.copy()]


        current_rmsprop = start_point.copy()
        current_adam = start_point.copy()



        for step in range(1000):
            grad_rmsprop = rosenbrock_gradient(current_rmsprop[0], current_rmsprop[1])

            current_rmsprop = rmsprop.update(current_rmsprop, grad_rmsprop)

            points_rmsprop.append(current_rmsprop.copy())



            grad_adam = rosenbrock_gradient(current_adam[0], current_adam[1])
            current_adam = adam.update(current_adam, grad_adam)
            points_adam.append(current_adam.copy())



            if step % 200 == 0:

                loss_rmsprop = rosenbrock_function(current_rmsprop[0], current_rmsprop[1])


                loss_adam = rosenbrock_function(current_adam[0], current_adam[1])

                print(f"Step {step:4d}: RMSProp loss = {loss_rmsprop:.6f}, adam loss = {loss_adam:.6f}")


        return points_rmsprop, points_adam