import gymnasium as gym
import numpy as np

# These environments will use Numpy.
# Will only convert to PyTorch when we need to.
class Terrain:
    def __init__(self):
        pass

    # Calculates heatmap features at (x, y) coordinate
    def features(self, x, y):
        return np.random.randn(10)

class CollaborativeEnv(gym.Env):
    def __init__(self, terrain: Terrain):
        pass
