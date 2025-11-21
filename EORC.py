import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

class EOReservoir:
    """
    Electro-Optic Delay-based Reservoir Computer (EO-RC) following the
    simplest model in D. Brunner et al., J. Appl. Phys. 124, 152004 (2018),
    involving a single Mach-Zhender interferometer, a gain element, and a
    delay line.
    
    """

    def __init__(self, res_size, alpha=0.8, beta=0.25, phi=0.2, seed=None):
        """

        Initialize the Echo State Network.

        Parameters
            res_size (int): Number of resevoir nodes
            alpha (int): feedback gain
            beta (int): input scaling
            phi (int): Mach-Zhender bias phase
            seed (int): Random seed
       
        """
        # RNG
        self.rng = np.random.default_rng(seed)

        # Stores size, and device parameters
        self.res_size = res_size
        self.alpha = alpha
        self.beta  = beta
        self.phi   = phi

        # Initial reservoir state
        self.x = np.zeros(res_size)