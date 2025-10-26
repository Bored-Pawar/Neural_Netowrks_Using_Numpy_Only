from layer import Layer
try:
    import cupy as np  # Use CuPy for GPU acceleration
except ImportError:
    import numpy as np  # Fallback to NumPy if CuPy not available

"""
NOTE: This is a batch-processing version of the Softmax layer.

DIFFERENCE FROM softmax.py:
- Original softmax.py: Applies softmax on axis=0, designed for single samples
- This file: Applies softmax on axis=1 (class dimension) for batches

INPUT FORMAT:
- From DenseBatch: (batch_size, num_classes, 1)
- Softmax is applied across num_classes for each sample independently

USAGE:
- Use this SoftmaxBatch in CNN architectures (VGG, Inception, etc.)
- Use original Softmax for single-sample networks
"""

class SoftmaxBatch(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_batch):
        """
        Forward pass: Apply softmax to batch

        Args:
            input_batch: Shape (batch_size, num_classes, 1)

        Returns:
            output: Shape (batch_size, num_classes, 1)
        """
        self.input = input_batch

        # Subtract max for numerical stability (along class dimension, axis=1)
        # Keep dims to allow broadcasting
        stable_input = input_batch - np.max(input_batch, axis=1, keepdims=True)

        # Compute exponential
        exponential_term = np.exp(stable_input)

        # Normalize by sum (along class dimension, axis=1)
        self.output = exponential_term / np.sum(exponential_term, axis=1, keepdims=True)

        return self.output

    def backward(self, output_gradient):
        """
        Backward pass for Softmax

        When using Softmax with CCE loss, the gradient simplifies to (y_pred - y_true)
        This is already handled in CCE_prime, so we just pass the gradient through

        Args:
            output_gradient: Shape (batch_size, num_classes, 1)

        Returns:
            input_gradient: Shape (batch_size, num_classes, 1)
        """
        # For Softmax + CCE, the backward pass is already simplified in the loss function
        # Just pass through the gradient
        return output_gradient
