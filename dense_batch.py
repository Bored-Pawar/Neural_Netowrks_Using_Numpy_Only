from layer import Layer
try:
    import cupy as np  # Use CuPy for GPU acceleration
except ImportError:
    import numpy as np  # Fallback to NumPy if CuPy not available

"""
NOTE: This is a batch-processing version of the Dense layer.

DIFFERENCE FROM dense.py:
- Original dense.py processes ONE sample at a time: input shape (features, 1)
- This file processes BATCHES of samples: input shape (batch_size, features, 1)

WHY NEEDED:
- CNN layers naturally output batches: (batch_size, height, width, channels)
- After Flatten: (batch_size, features, 1)
- Need Dense layer that can handle entire batches for efficiency

KEY CHANGES:
1. Forward pass: Loop over batch and process each sample individually
2. Backward pass: Accumulate gradients from all samples in the batch
3. Gradient averaging: Divide by batch_size to get average gradient

USAGE:
- Use this DenseBatch in CNN architectures (VGG, Inception, etc.)
- Use original Dense for single-sample networks (XOR, Two Moons, etc.)
"""

class DenseBatch(Layer):
    def __init__(self, input_size, output_size):
        """
        Dense (Fully Connected) layer with batch processing support

        Args:
            input_size: Number of input features
            output_size: Number of output features (neurons)
        """
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

        self.weight_gradient = None
        self.bias_gradient = None

        # For Adam optimizer
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)

    def forward(self, input_batch):
        """
        Forward pass with batch processing

        Args:
            input_batch: Shape (batch_size, input_size, 1)

        Returns:
            output_batch: Shape (batch_size, output_size, 1)
        """
        self.input = input_batch
        batch_size = input_batch.shape[0]

        # Initialize output for entire batch
        output_batch = np.zeros((batch_size, self.weights.shape[0], 1))

        # Process each sample in the batch
        for i in range(batch_size):
            sample = input_batch[i]  # Shape: (input_size, 1)
            output_batch[i] = np.dot(self.weights, sample) + self.bias

        return output_batch

    def backward(self, output_gradient_batch):
        """
        Backward pass with batch processing

        Args:
            output_gradient_batch: Shape (batch_size, output_size, 1)

        Returns:
            input_gradient_batch: Shape (batch_size, input_size, 1)
        """
        batch_size = output_gradient_batch.shape[0]

        # Initialize gradients
        self.weight_gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.bias)
        input_gradient_batch = np.zeros_like(self.input)

        # Accumulate gradients from all samples in the batch
        for i in range(batch_size):
            output_grad = output_gradient_batch[i]  # Shape: (output_size, 1)
            sample_input = self.input[i]  # Shape: (input_size, 1)

            # Accumulate weight gradient: dE/dW = dE/dY * X^T
            self.weight_gradient += np.dot(output_grad, sample_input.T)

            # Accumulate bias gradient: dE/db = dE/dY
            self.bias_gradient += output_grad

            # Compute input gradient for this sample: dE/dX = W^T * dE/dY
            input_gradient_batch[i] = np.dot(self.weights.T, output_grad)

        # Average gradients over the batch
        # This is important for stable training with different batch sizes
        self.weight_gradient /= batch_size
        self.bias_gradient /= batch_size

        return input_gradient_batch
