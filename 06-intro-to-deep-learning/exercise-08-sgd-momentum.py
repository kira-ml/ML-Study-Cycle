import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)




def generate_separable_data(n_sample=1000):


    class_0 = np.random.multivariate_normal(
        mean=[-1, 1], cov=[[1, 0], [0, 1]] 
    )