# all necessary layer, activation fuction, optimizer and losses are imported from respective files
from dense import Dense
from activation import ReLU, Sigmoid, Tanh
from adam import Adam
from losses import MSE, MSE_prime

# import numpy and scikit learn
import numpy as np
from sklearn.datasets import make_moons

# data generation
X, Y = make_moons(n_samples=500, noise=0.1, random_state=42)
X = X.reshape(len(X), 2, 1)
Y = Y.reshape(len(Y), 1, 1)

# network design
network = [
    Dense(2, 16), 
    ReLU(),
    Dense(16, 1), 
    Sigmoid()    
]

epochs = 10000
learning_rate = 0.001

# Create the optimizer
optimizer = Adam(learning_rate=learning_rate)

# training loop 
for epoch in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        # Forward pass
        output = x
        for layer in network:
            output = layer.forward(output)
        
        # Compute loss 
        error += MSE(y, output)
        
        # Backward pass 
        output_gradient = MSE_prime(y, output)
        for layer in reversed(network):
            output_gradient = layer.backward(output_gradient)
            
            if isinstance(layer, Dense):
                optimizer.update(layer)
    
    error /= len(X)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{epochs}, Error: {error}')