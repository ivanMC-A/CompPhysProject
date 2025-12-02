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
        
        # Placeholders for network properties
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
            u_train (ndarray): training input
            y_train (ndarray): training output
            method (str): "ridge" or "ols"
            reg (float): ridge parameter
            train_skip (int): Steps to skip for training
        """
        
        if u_train.shape[0] > 1 or y_train.shape[0] > 1:
            raise ValueError("Data must be reshapped!")

        #Determine input size
        self.in_size = u_train.shape[0]

        #Determine output size
        self.out_size = y_train.shape[0]

        # Initialize input weight matrix
        self.w_in = self.rng.random((self.res_size, self.in_size + 1)) - 0.5

        # Initialize reservoir state
        x = np.zeros((self.res_size, 1))

        # Reset saved states
        self.x = np.zeros((self.res_size + 1, u_train.shape[1] - train_skip)) + 1

        # Set states
        for i in range(u_train.shape[1]):
            u = u_train[:, i]
            x = np.tanh(np.dot(self.w_in, np.vstack((1, u))) + np.dot(self.w_res, x))

            if i>= train_skip:
                self.x[:, i - train_skip] = np.vstack((1, x))[:, 0]

        # Train output weights
        if method == "ridge":
            self.w_out = np.linalg.solve(
                np.dot(self.x, self.x.T) + reg * np.eye(1 + self.res_size),
                np.dot(self.x, y_train[:, train_skip:].T)
            ).T
        elif method == "ols":
            self.w_out = np.linalg.solve(
                np.dot(self.x, self.x.T),
                np.dot(self.x, y_train[:, train_skip:].T)
            ).T
        else:
            raise ValueError("Unsupported training method")
        
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
        x = self.x[1:, self.x.shape[1] - 1:]

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