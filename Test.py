import Tokeniser as tk
import numpy as np


t = tk.SimpleTokeniser()
samples = {'Jupiter has 79 known moons .', 'Neptune has 14 confirmed moons !'}
print(t.fromDataset(samples))