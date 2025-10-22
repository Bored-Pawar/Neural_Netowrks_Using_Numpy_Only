from layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

        self.weight_gradient = None
        self.bias_gradient =  None

        # for adam
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient):
        self.weight_gradient = np.dot(output_gradient, self.input.T) # del E / del W   (gradient)
        self.bias_gradient =  output_gradient # gradient decent of bias

        # removed the change of weight now adam will do it
        # self.weights -= learning_rate * self.weight_gradient # gradient decent of weight

        return np.dot(self.weights.T, output_gradient) # returning del E / del X which will be del E / del Y of previous layer