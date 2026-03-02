import numpy as np

class Encoder:

    def hotOneEncoding(self, tokens: dict[str, int]):
        return np.eye(2)[len(tokens)]