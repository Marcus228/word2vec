import numpy as np
def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))