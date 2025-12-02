import numpy as np

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

        # Store size system parameters
        self.res_size = res_size

        # Create randomized matrix network
        self.w_res = self.rng.random((self.res_size, self.res_size)) - 0.5

        # Normalize matrix network
        self.w_res *= 1 / np.max(np.abs(np.linalg.eigvals(self.w_res)))
        
        # Placeholders for input/output sizes and weights
        self.in_size  = None
        self.out_size = None
        self.w_in     = None
        self.w_out    = None
        self.x        = None

    def fit(self, u_train, y_train, method = "ridge", reg=1e-6, train_skip = 0):
        """
        This is the training function for the ESN. Allowing for control
        over fit start point and fitting type.

        Parameters
            u_train (ndarray): training input sequence (T x 1)
            y_train (ndarray): training target sequence (T x output_dim)
            method (str): "ridge" or "ols"
            reg (float): ridge parameter
            train_skip (int): washout period
        """

        #Determine input size
        if u_train.ndim == 2:
            self.in_size = u_train.shape[1]
            T = u_train.shape[0]
        else:
            self.in_size = 1
            T = len(u_train)

        #Determine output size
        if y_train.ndim == 2:
            self.out_size = y_train.shape[1]
        else:
            self.out_size = 1

        # Initialize input weight matrix
        self.w_in = self.rng.random((self.res_size, self.in_size + 1)) - 0.5

        # Initialize reservoir state
        x = np.zeros((self.res_size, 1))

        # Collect states
        X = np.zeros((self.res_size + 1, T))
        for t in range(T):
            
            if self.in_size > 1:
                u = u_train[t, :].reshape(-1, 1)
            else:
                u = np.array([[u_train[t]]])

            x = np.tanh(np.dot(self.w_in, np.vstack((1, u))) + np.dot(self.w_res, x))
            X[1:, t] = x[:, 0]
        X[0, :] = 1

        # Apply washout
        x_used = X[:, train_skip:]
        if y_train.ndim == 2:
            y_used = y_train[train_skip:, :].T
        else:
            y_used = y_train[train_skip:].reshape(1, -1)

        # Save state matrix
        self.x = X

        # Train output weights
        if method == "ridge":
            self.w_out = np.dot(y_used, np.dot(x_used.T, np.linalg.inv(np.dot(x_used, x_used.T) + reg * np.eye(self.res_size + 1))))
        elif method == "ols":
            self.w_out = np.dot(y_used, np.linalg.pinv(x_used))
        else:
            raise ValueError("Unsupported training method")

        return self.w_out


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
        x = self.x[1:, -1:].reshape(-1, 1)

        #Initialize input 
        u = u_init.reshape(-1, 1)

        # Closed-loop prediction
        for i in range(test_length):
            
            # Update resevoir state
            x = np.tanh(np.dot(self.w_in, np.vstack((1, u))) + np.dot(self.w_res, x))
            
            # Compute output
            y = np.dot(self.w_out, np.vstack((1, x)))
            
            # Store output
            prediction[:, i] = y[:, 0]

            # Feedbacks
            u = y

        return prediction