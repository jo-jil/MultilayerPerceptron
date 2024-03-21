
'''
Emre Aktas & Joseph Tschopp
maktas2@u.rochester.edu & jtschopp@u.rochester.edu 
CSC 246 - HW 4: Classification
03/17/2024
'''

import numpy as np
import pickle
from util import softmax

class MultilayerPerceptron:
    # Constructor with default hyperparameters
    def __init__(self, learning_rate=0.01, epochs=1000, hidden_size=10, initialization='he'):
        # Learning rate for gradient descent
        self.learning_rate = learning_rate
        # Number of passes through the dataset
        self.epochs = epochs
        # Number of neurons in the hidden layer
        self.hidden_size = hidden_size
        # Method for initializing weights ('zeros', 'random', 'xavier', 'he')
        self.initialization = initialization
        # Weights from input layer to hidden layer
        self.weights_input_to_hidden = None
        # Weights from hidden layer to output layer
        self.weights_hidden_to_output = None
        # Bias values for hidden layer
        self.bias_hidden = None
        # Bias values for output layer
        self.bias_output = None
        # Mean of features in training dataset for normalization
        self.X_mean = None
        # Standard deviation of features in training dataset for normalization
        self.X_std = None

    # Activation function for neurons in the hidden layer
    def _tanh(self, x):
        return np.tanh(x)

    # Derivative of tanh function, used in backpropagation
    def _tanh_derivative(self, x):
        return 1 - np.tanh(x)**2

    # Initializes weights according to the specified method
    def _initialize_weights(self, input_size, output_size):
        if self.initialization == 'zeros':
            # Initialize weights to zeros
            self.weights_input_to_hidden = np.zeros((input_size, self.hidden_size))
            self.weights_hidden_to_output = np.zeros((self.hidden_size, output_size))
        elif self.initialization == 'random':
            # Initialize weights to small random values
            self.weights_input_to_hidden = np.random.randn(input_size, self.hidden_size) * 0.01
            self.weights_hidden_to_output = np.random.randn(self.hidden_size, output_size) * 0.01
        elif self.initialization == 'xavier':
            # Xavier/Glorot initialization
            bound = np.sqrt(6 / (input_size + output_size))
            self.weights_input_to_hidden = np.random.uniform(-bound, bound, (input_size, self.hidden_size))
            self.weights_hidden_to_output = np.random.uniform(-bound, bound, (self.hidden_size, output_size))
        elif self.initialization == 'he':
            # He initialization for ReLU networks
            self.weights_input_to_hidden = np.random.randn(input_size, self.hidden_size) * np.sqrt(2 / input_size)
            self.weights_hidden_to_output = np.random.randn(self.hidden_size, output_size) * np.sqrt(2 / self.hidden_size)
        else:
            raise ValueError("Unknown initialization method")

    # Trains the MLP using the provided dataset
    def fit(self, X, T):
        # Normalize the input features
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        X = (X - self.X_mean) / self.X_std

        # Determine the size of the input and output layers
        input_size = X.shape[1]
        output_size = T.shape[1]

        # Initialize weights and biases
        self._initialize_weights(input_size, output_size)
        self.bias_hidden = np.zeros(self.hidden_size)
        self.bias_output = np.zeros(output_size)

        # Training loop
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # Forward pass
                xi = X[i:i+1]  # Input for current sample
                ti = T[i:i+1]  # Target for current sample

                # Compute activations for hidden and output layers
                hidden_layer_input = np.dot(xi, self.weights_input_to_hidden) + self.bias_hidden
                hidden_layer_output = self._tanh(hidden_layer_input)
                output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_to_output) + self.bias_output
                output_layer_output = softmax(output_layer_input)

                # Backpropagation to compute gradients
                output_error = output_layer_output - ti
                hidden_error = np.dot(output_error, self.weights_hidden_to_output.T) * self._tanh_derivative(hidden_layer_input)

                # Update weights and biases using gradient descent
                self.weights_hidden_to_output -= self.learning_rate * np.dot(hidden_layer_output.T, output_error)
                self.weights_input_to_hidden -= self.learning_rate * np.dot(xi.T, hidden_error)
                self.bias_output -= self.learning_rate * np.sum(output_error, axis=0)
                self.bias_hidden -= self.learning_rate * np.sum(hidden_error, axis=0)

    # Predicts the class labels for the given inputs
    def predict(self, X):
        # Normalize the input features
        X = (X - self.X_mean) / self.X_std

        # Forward pass to compute the model output
        hidden_layer_input = np.dot(X, self.weights_input_to_hidden) + self.bias_hidden
        hidden_layer_output = self._tanh(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_to_output) + self.bias_output
        output_layer_output = softmax(output_layer_input)
        
        # Return the class with the highest probability
        return output_layer_output

    # Saves the model parameters to a file
    def save(self, filename):
        if not filename.endswith('.model'):
            filename += '.model'
        with open(filename, 'wb') as file:
            pickle.dump({
                'weights_input_to_hidden': self.weights_input_to_hidden,
                'weights_hidden_to_output': self.weights_hidden_to_output,
                'bias_hidden': self.bias_hidden,
                'bias_output': self.bias_output,
                'X_mean': self.X_mean,  # Save normalization parameters
                'X_std': self.X_std
            }, file)

    # Loads the model parameters from a file
    @staticmethod
    def load(filename):
        if not filename.endswith('.model'):
            filename += '.model'
        with open(filename, 'rb') as file:
            model_params = pickle.load(file)
        model = MultilayerPerceptron()
        model.weights_input_to_hidden = model_params['weights_input_to_hidden']
        model.weights_hidden_to_output = model_params['weights_hidden_to_output']
        model.bias_hidden = model_params['bias_hidden']
        model.bias_output = model_params['bias_output']
        # Load normalization parameters
        model.X_mean = model_params.get('X_mean', None)
        model.X_std = model_params.get('X_std', None)
        return model