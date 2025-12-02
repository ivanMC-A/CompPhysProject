import numpy as np

class EO_RC:
    """
    A class for creating an EO-Reservoir Computer (EO-RC).
    """

    #Initialize a new EO-RC
    def __init__(self, res_size, alpha=0.8, beta=0.25, phi=0.2, seed=None):
        """
        Initialize the Echo State Network.

        Parameters
            res_size (int): Number of resevoir nodes
            alpha (float): Feedback gain
            beta (float): Input scaling
            phi (float): Mach-Zhender bias phase
            seed (int): Random seed
        """

        # RNG
        self.rng = np.random.default_rng(seed)

        # Stores size and device parameters
        self.res_size = res_size
        self.alpha = alpha
        self.beta  = beta
        self.phi   = phi

        # Initial reservoir state
        self.x = np.zeros(res_size)

        # Placeholders for training-derived values
        self.in_size  = None
        self.out_size = None
        self.w_out    = None
        self.x        = None
        self.y        = None


    def _nonlinearity(self, z):
        """
        Mach-Zehnder-like sinusoidal nonlinearity.
        
        Parameters
            z (float): Current phase from feedback of previous nodes and input signal
        """
        return np.sin(z + self.phi)


    def _step(self, u):
        """
        Update the reservoir using virtual node time-multiplexing

            Parameters
                u (float): Input signal
        """
        # Initialize current timestep
        new_x = np.zeros(self.res_size)

        # Fill nodes based off previous timestep
        new_x[0] = self._nonlinearity(self.alpha * self.x[-1] + self.beta * u)
        for i in range(1, self.res_size):
            new_x[i] = self._nonlinearity(self.alpha * new_x[i-1] + self.beta * u)

        self.x = new_x
        return new_x


    def fit(self, u_train, y_train, method="ridge", reg=1e-6, train_skip=0):
        """
        Train the EO-Reservoir readout layer.

        Parameters
            u_train (ndarray): training input sequence (T x 1)
            y_train (ndarray): training target sequence (T x output_dim)
            method (str): "ridge" or "ols"
            reg (float): ridge parameter
            train_skip (int): washout period
        """

        # Detect input/output size
        self.in_size  = u_train.shape[1] if u_train.ndim == 2 else 1
        self.out_size = y_train.shape[1] if y_train.ndim == 2 else 1

        T = len(u_train)

        # Store states
        X = np.zeros((self.res_size, T))
        Y = y_train.T  # shape: (out_size, T)

        # Reset reservoir
        self.x = np.zeros(self.res_size)

        # Collect states
        for t in range(T):
            u = float(u_train[t])
            x_state = self._step(u)
            X[:, t] = x_state

        # Add bias term
        X_aug = np.vstack((np.ones(T), X))

        # Save state matrix for prediction
        self.x = X_aug

        # Train output weights
        if method == "ridge":
            self.w_out = np.dot(np.dot(Y, X_aug.T), np.linalg.inv(np.dot(X_aug, X_aug.T) + reg * np.eye(self.res_size + 1)))
        elif method == "ols":
            self.w_out = np.dot(Y, np.linalg.pinv(X_aug))
        else:
            raise ValueError("Unsupported training method")

        return self.w_out


    def predict(self, u_init, test_length):
        """
        Generate closed-loop predictions using the trained EO-Resevoir Computer (EO-RC).

        Parameters
            u_init (ndarray): Initial input vector used to seed the prediction loop.
            test_length (int): Number of time steps to predict.

        Returns
            prediction (ndarray): Array containing the predicted outputs.
        """

        # Initialize output
        prediction = np.zeros((self.out_size, test_length))

        # Trained resevoir state
        self.x = self.x[1:, -1:].reshape(-1)

        #Initialize input 
        if np.isscalar(u_init):
            u = float(u_init)
        else:
            u= float(u_init[0])

        # Closed-loop autonomous prediction
        for i in range(test_length):

            # Reservoir update via EO time-multiplexed dynamics
            new_x = np.zeros(self.res_size)

            # First virtual node uses last node from previous timestep
            new_x[0] = self._nonlinearity(self.alpha * self.x[-1] + self.beta * u)

            # Remaining nodes depend on previous virtual node
            for j in range(1, self.res_size):
                new_x[j] = self._nonlinearity(self.alpha * new_x[j - 1] + self.beta * u)

            # Commit reservoir update
            self.x = new_x

            # Compute output-
            y = np.dot(self.w_out, np.concatenate(([1], self.x)))

            # Store prediction
            prediction[:, i] = y

            # Feedback: output becomes next input
            u = y

        return prediction