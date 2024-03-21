import numpy as np
import pickle
from scipy.stats import zscore
from util import *

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.val_accuracies = []
        # Store the mean and standard deviation for normalization during prediction
        self.X_mean = None
        self.X_std = None

    def fit(self, X, T, X_val=None, T_val=None):
        # Compute the mean and standard deviation for the training data
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)

        # Apply Z-score normalization to the training data
        X = (X - self.X_mean) / self.X_std

        # If validation data is provided, normalize it using training data statistics
        if X_val is not None:
            X_val = (X_val - self.X_mean) / self.X_std

        n_samples, n_features = X.shape
        n_outputs = T.shape[1]
        self.weights = np.random.randn(n_features, n_outputs)
        self.bias = np.zeros(n_outputs)

        for epoch in range(self.epochs):
            outputs = np.dot(X, self.weights) + self.bias
            y_pred = softmax(outputs)  # Use softmax from util.py
            error = y_pred - T

            self.weights -= self.learning_rate * np.dot(X.T, error) / n_samples
            self.bias -= self.learning_rate * np.sum(error, axis=0) / n_samples

            # Calculate and store validation accuracy after each epoch
            if X_val is not None and T_val is not None:
                val_predictions = self.predict(X_val)
                val_accuracy = np.mean(np.argmax(T_val, axis=1) == val_predictions)
                self.val_accuracies.append(val_accuracy)
    
    def predict(self, X):
        # Apply the same normalization as was applied to the training data
        X = (X - self.X_mean) / self.X_std
        
        z = np.dot(X, self.weights) + self.bias
        probabilities = softmax(z)  # Use softmax from util.py
        return probabilities



    def save(self, filename):
        # Ensure the filename ends with '.model'
        if not filename.endswith('.model'):
            filename += '.model'
        # Save model parameters with pickle
        with open(filename, 'wb') as file:
            pickle.dump({
                'weights': self.weights,
                'bias': self.bias,
                'X_mean': self.X_mean,  # Also save the normalization parameters
                'X_std': self.X_std
            }, file)

    @staticmethod
    def load(filename):
        # Ensure the filename ends with '.model'
        if not filename.endswith('.model'):
            filename += '.model'
        with open(filename, 'rb') as file:
            model_params = pickle.load(file)
        model = LogisticRegression()
        model.weights = model_params['weights']
        model.bias = model_params['bias']
        # Load the normalization parameters
        model.X_mean = model_params.get('X_mean', None)
        model.X_std = model_params.get('X_std', None)
        return model