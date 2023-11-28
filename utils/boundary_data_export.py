import numpy as np

data = np.load("0.2.npy")


class ParticleDataExport:
    def __init__(self, data):
        self.data = data.reshape(-1, 4, 4)
        for item in self.data:
            item[0, :3] *= 1000  # m -> mm

