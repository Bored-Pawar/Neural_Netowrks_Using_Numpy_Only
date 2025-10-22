import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient):
        # TODO: set the weights
        pass

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

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient):
        return np.multiply(output_gradient, self.activation_prime(self.input)) # returning the derivative using simplified equation
    
class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))
        super().__init__(sigmoid, sigmoid_prime)

class ReLU(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: (x > 0).astype(int)
        super().__init__(relu, relu_prime)

def MSE(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def MSE_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

X = np.reshape(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (4, 2, 1))
Y = np.reshape(np.array([[0], [1], [1], [0]]), (4, 1, 1))

network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

epochs = 100000
# Note: Adam often works better with a smaller learning rate
learning_rate = 0.001 

# create the optimizer
optimizer = Adam(learning_rate=learning_rate)

# Training Loop
for epoch in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        # Forward pass (no changes here)
        output = x
        for layer in network:
            output = layer.forward(output)
        
        # Compute loss 
        error += MSE(y, output)
        
        # Backward pass 
        output_gradient = MSE_prime(y, output)
        for layer in reversed(network):
            # Pass the gradient back through the layer
            output_gradient = layer.backward(output_gradient)
            
            # 2. If the layer is trainable, tell the optimizer to update it
            if isinstance(layer, Dense):
                optimizer.update(layer)
    
    error /= len(X)
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Error: {error}')