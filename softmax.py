from layer import Layer
import numpy as np

class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # we subtract max rom each input so that the equation to get softmax becomes stable
        stable_input = input - np.max(input, axis = 0)
        exponential_term = np.exp(stable_input)

        self.output = exponential_term / np.sum(exponential_term, axis = 0)
        return self.output
    
