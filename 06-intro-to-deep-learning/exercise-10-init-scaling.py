import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable
import seaborn as sns




plt.style.use('seaborn-v0_8')

np.random.seed(42)





def xavier_initialization(fan_in: int, fan_out: int) -> np.ndarray:

    scale = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0, scale, size=(fan_in, fan_out))



def he_initialization(fan_in: int, fan_out: int) -> np.ndarray:



    scale = np.sqrt(2.0 / fan_in)

    return np.random.normal(0, scale, size=(fan_in, fan_out))



def uniform_xavier(fan_in: int, fan_out: int) -> np.ndarray:



    limit =  np.sqrt(6.0 / (fan_in + fan_out))



    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))