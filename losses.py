import numpy as np

# mean square error
def MSE(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def MSE_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

# binary cross entropy
def BCE(y_true, y_pred):
    # add a small epsilon to prevent log(0) errors
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def BCE_prime(y_true, y_pred):
    # Add a small epsilon to prevent division by zero
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -((y_true / y_pred) - ((1 - y_true) / (1 - y_pred))) / np.size(y_true)

# softmax
# to be made