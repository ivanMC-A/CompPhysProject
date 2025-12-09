import numpy as np

class EORC:
    """
    A class for creating an EO-Reservoir Computer (EO-RC).
    """

    #Initialize a new EO-RC
    def __init__(self, res_size, alpha=0.8, beta=0.25, phi=0.2, seed=None):
        """
        Initialize the EO-Resevoir.

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

        
        # Placeholders for network properties
        self.in_size  = None
        self.out_size = None
        self.w_out    = None
        self.x        = None

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

        if u_train.shape[0] > 1 or y_train.shape[0] > 1:
            raise ValueError("Data must be reshapped!")

        #Determine input size
        self.in_size = u_train.shape[0]

        #Determine output size
        self.out_size = y_train.shape[0]

        # Initialize input weight matrix
        self.w_in = self.rng.random((self.res_size, self.in_size + 1)) * self.alpha

        # Initialize reservoir state
        x = np.zeros((self.res_size, 1))

        # Reset saved states
        self.x = np.zeros((self.res_size + 1, u_train.shape[1] - train_skip)) + 1

        # Set states
        for i in range(u_train.shape[1]):
            u = u_train[:, i]
            x = np.sin(np.dot(self.w_in, np.vstack((1, u))) + self.beta*x + self.phi)**2

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
        x = self.x[1:, self.x.shape[1] - 1:]

        #Initialize input 
        u = u_init

        # Closed-loop prediction
        for i in range(test_length):
            
            # Update resevoir state
            x = np.sin(np.dot(self.w_in, np.vstack((1, u))) + self.beta*x + self.phi)**2
            
            # Compute output
            y = np.dot(self.w_out, np.vstack((1, x)))
            
            # Store output
            prediction[:, i] = y[:, 0]

            # Feedback
            u = y

        return prediction
    
    def score(self, u_init, y, test_length):
        """
        Returns the mean squared error of the model.
        """
        return np.mean((self.predict(u_init,test_length) - y)**2,axis =0)