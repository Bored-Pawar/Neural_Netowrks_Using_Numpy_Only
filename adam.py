try:
    import cupy as np  # Use CuPy for GPU acceleration
except ImportError:
    import numpy as np  # Fallback to NumPy if CuPy not available

class Adam():
    def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0 # time step counter
    
    def update(self, layer):
        self.t += 1 # increment time step for each update


        # Update weights

        # step 1 >> update momentum >> m
        layer.m_weights = self.beta1 * layer.m_weights + (1 - self.beta1) * layer.weight_gradient 

        # step 2 >> update velocity >> v
        layer.v_weights = self.beta2 * layer.v_weights + (1 - self.beta2) * (layer.weight_gradient ** 2)

        # step 3 >> bias correction
        m_corrected = layer.m_weights / (1 - self.beta1 ** self.t)
        v_corrected = layer.v_weights / (1 - self.beta2 ** self.t)

        # step 4 >> final weight update
        layer.weights -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)


        # Update biases

        # step 1 >> update momentum >> m
        layer.m_bias = self.beta1 * layer.m_bias + (1 - self.beta1) * layer.bias_gradient 

        # step 2 >> update velocity >> v
        layer.v_bias = self.beta2 * layer.v_bias + (1 - self.beta2) * (layer.bias_gradient ** 2)

        # step 3 >> bias correction
        m_corrected = layer.m_bias / (1 - self.beta1 ** self.t)
        v_corrected = layer.v_bias / (1 - self.beta2 ** self.t)

        # step 4 >> final bias update
        layer.bias -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)

class AdamCNN():
    def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0 # time step counter

    def update(self, layer):
        self.t += 1 # increment time step for each update

        # Update filters (weights for conv layers)

        # step 1 >> update momentum >> m
        layer.m_filters = self.beta1 * layer.m_filters + (1 - self.beta1) * layer.filters_gradient

        # step 2 >> update velocity >> v
        layer.v_filters = self.beta2 * layer.v_filters + (1 - self.beta2) * (layer.filters_gradient ** 2)

        # step 3 >> bias correction
        m_corrected = layer.m_filters / (1 - self.beta1 ** self.t)
        v_corrected = layer.v_filters / (1 - self.beta2 ** self.t)

        # step 4 >> final filter update
        layer.filters -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)


        # Update biases

        # step 1 >> update momentum >> m
        layer.m_bias = self.beta1 * layer.m_bias + (1 - self.beta1) * layer.bias_gradient

        # step 2 >> update velocity >> v
        layer.v_bias = self.beta2 * layer.v_bias + (1 - self.beta2) * (layer.bias_gradient ** 2)

        # step 3 >> bias correction
        m_corrected = layer.m_bias / (1 - self.beta1 ** self.t)
        v_corrected = layer.v_bias / (1 - self.beta2 ** self.t)

        # step 4 >> final bias update
        layer.bias -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)