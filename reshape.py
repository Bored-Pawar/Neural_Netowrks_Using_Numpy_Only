from layer import Layer
try:
    import cupy as np  # Use CuPy for GPU acceleration
except ImportError:
    import numpy as np  # Fallback to NumPy if CuPy not available

class Flatten(Layer):
    def __init__(self):
        """
        Flatten layer to convert 4D tensor to 2D for Dense layers

        Converts (batch_size, height, width, channels) -> (batch_size, height*width*channels, 1)
        This format is compatible with the existing Dense layer implementation
        """
        self.input_shape = None

    def forward(self, input_data):
        """
        Forward pass: Flatten the input

        Args:
            input_data: Shape (batch_size, height, width, channels)

        Returns:
            output: Shape (batch_size, height*width*channels, 1)
        """
        self.input_shape = input_data.shape
        batch_size, height, width, channels = input_data.shape

        # Flatten to (batch_size, height*width*channels)
        flattened = input_data.reshape(batch_size, height * width * channels)

        # Add the extra dimension to match Dense layer format: (batch_size, features, 1)
        output = flattened.reshape(batch_size, height * width * channels, 1)

        return output

    def backward(self, output_gradient):
        """
        Backward pass: Reshape gradient back to original input shape

        Args:
            output_gradient: Shape (batch_size, height*width*channels, 1)

        Returns:
            input_gradient: Shape (batch_size, height, width, channels)
        """
        batch_size, height, width, channels = self.input_shape

        # Remove the extra dimension and reshape back
        # (batch_size, features, 1) -> (batch_size, features) -> (batch_size, h, w, c)
        input_gradient = output_gradient.reshape(batch_size, height * width * channels)
        input_gradient = input_gradient.reshape(batch_size, height, width, channels)

        return input_gradient


class Reshape(Layer):
    def __init__(self, target_shape):
        """
        General reshape layer

        Args:
            target_shape: Tuple of target dimensions (excluding batch size)
                         e.g., (7, 7, 64) or (128, 1) for Dense layer format
        """
        self.target_shape = target_shape
        self.input_shape = None

    def forward(self, input_data):
        """
        Forward pass: Reshape the input

        Args:
            input_data: Any shape starting with batch_size

        Returns:
            output: Shape (batch_size, *target_shape)
        """
        self.input_shape = input_data.shape
        batch_size = input_data.shape[0]

        # Reshape keeping batch size as first dimension
        output = input_data.reshape(batch_size, *self.target_shape)

        return output

    def backward(self, output_gradient):
        """
        Backward pass: Reshape gradient back to original input shape

        Args:
            output_gradient: Shape (batch_size, *target_shape)

        Returns:
            input_gradient: Original input shape
        """
        # Reshape back to original input shape
        input_gradient = output_gradient.reshape(self.input_shape)

        return input_gradient
