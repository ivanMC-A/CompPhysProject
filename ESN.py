import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

class ESN:
    """
    A class for creating an basic echo state network (ESN).

    """

    #Initialize a new ESN
    def __init__(self, res_size, seed=None):
        """
        Initialize the Echo State Network.

        Parameters
            res_size (int): Number of resevoir nodes
            seed (int): Random seed

        """

        # RNG
        self.rng = np.random.default_rng(seed)

        # Store size
        self.res_size = res_size

        # Create randomized matrix network
        self.w_res = self.rng.random((self.res_size, self.res_size)) - 0.5

        # Normalize matrix network
        self.w_res *= 1 / max(abs(np.linalg.eigvals(self.w_res)))

    def fit(self, u_train, y_train, method = "ridge", train_skip = 0):
        """
        This is the training function for the ESN. Allowing for control
        over fit start point and fitting type.

        Methods
        "ridge", "ols", "lasso", "elastic", "sgd", "rls"
        """

    def predict(self, u_init, test_length):
        """
        Generate closed-loop predictions using the trained Echo State Network (ESN).

        Parameters
            u_init (ndarray): Initial input vector used to seed the prediction loop.
            test_length (int): Number of time steps to predict.

        Returns
            prediction (ndarray): Array containing the predicted outputs.
        
        """

        # Initialize output
        prediction = np.zeros((self.out_size, test_length))

        # Trained resevoir state
        x = self.X[1:, -1:].reshape(-1, 1)

        #Initialize input 
        u = u_init

        # Closed-loop prediction
        for i in range(test_length):
            
            # Update resevoir state
            x = np.tanh(np.dot(self.w_in, np.vstack((1, u))) + np.dot(self.w_res, x))
            
            # Compute output
            y = np.dot(self.w_out, np.vstack((1, x)))
            
            # Store output
            prediction[:, i] = y[:, 0]

            # Feedback
            u = y

        return prediction