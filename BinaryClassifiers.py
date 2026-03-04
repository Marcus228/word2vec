import numpy as np
def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))