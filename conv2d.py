from layer import Layer
try:
    import cupy as np  # Use CuPy for GPU acceleration
    GPU_AVAILABLE = True
except ImportError:
    import numpy as np  # Fallback to NumPy if CuPy not available
    GPU_AVAILABLE = False
    print("CuPy not found, using NumPy (CPU). Install CuPy for GPU acceleration.")

class Conv2D(Layer):
    def __init__(self, num_filters, kernel_size, input_channels, stride=1, padding='valid'):
        """
        Initialize a 2D Convolutional Layer

        Args:
            num_filters: Number of filters (output channels)
            kernel_size: Size of the square kernel (e.g., 3 for 3x3)
            input_channels: Number of input channels
            stride: Stride for convolution (default=1)
            padding: 'valid' (no padding) or 'same' (zero padding)
        """
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.stride = stride
        self.padding = padding

        # Initialize filters with He initialization (good for ReLU)
        # Shape: (kernel_size, kernel_size, input_channels, num_filters)
        self.filters = np.random.randn(kernel_size, kernel_size, input_channels, num_filters) * np.sqrt(2.0 / (kernel_size * kernel_size * input_channels))

        # Initialize bias (one per filter)
        self.bias = np.zeros((num_filters, 1))

        # Gradients (will be computed during backward pass)
        self.filters_gradient = None
        self.bias_gradient = None

        # For Adam optimizer
        self.m_filters = np.zeros_like(self.filters)
        self.v_filters = np.zeros_like(self.filters)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)

    def _apply_padding(self, input_data):
        """Apply zero padding to input if padding='same'"""
        if self.padding == 'valid':
            return input_data, 0

        # Calculate padding needed for 'same'
        pad_size = (self.kernel_size - 1) // 2

        # Pad: (batch, height, width, channels)
        padded = np.pad(input_data,
                       ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                       mode='constant',
                       constant_values=0)
        return padded, pad_size

    def forward(self, input_data):
        """
        Forward pass of convolution

        Args:
            input_data: Shape (batch_size, height, width, channels)

        Returns:
            output: Shape (batch_size, out_height, out_width, num_filters)
        """
        self.input = input_data
        batch_size, input_height, input_width, _ = input_data.shape

        # Apply padding if needed
        padded_input, pad_size = self._apply_padding(input_data)
        _, padded_height, padded_width, _ = padded_input.shape

        # Calculate output dimensions
        out_height = (padded_height - self.kernel_size) // self.stride + 1
        out_width = (padded_width - self.kernel_size) // self.stride + 1

        # Initialize output
        output = np.zeros((batch_size, out_height, out_width, self.num_filters))

        # Perform convolution
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(out_height):
                    for j in range(out_width):
                        # Extract the region
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        region = padded_input[b, h_start:h_end, w_start:w_end, :]

                        # Convolve: element-wise multiply and sum
                        output[b, i, j, f] = np.sum(region * self.filters[:, :, :, f]) + self.bias[f]

        return output

    def backward(self, output_gradient):
        """
        Backward pass of convolution

        Args:
            output_gradient: Gradient from next layer, shape (batch_size, out_height, out_width, num_filters)

        Returns:
            input_gradient: Gradient w.r.t input, shape (batch_size, height, width, channels)
        """
        batch_size, input_height, input_width, _ = self.input.shape
        _, out_height, out_width, _ = output_gradient.shape

        # Apply padding to input
        padded_input, pad_size = self._apply_padding(self.input)
        _, padded_height, padded_width, _ = padded_input.shape

        # Initialize gradients
        self.filters_gradient = np.zeros_like(self.filters)
        self.bias_gradient = np.zeros_like(self.bias)
        padded_input_gradient = np.zeros_like(padded_input)

        # Compute gradients
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        # Gradient w.r.t filter
                        region = padded_input[b, h_start:h_end, w_start:w_end, :]
                        self.filters_gradient[:, :, :, f] += region * output_gradient[b, i, j, f]

                        # Gradient w.r.t input (backpropagate through the filter)
                        padded_input_gradient[b, h_start:h_end, w_start:w_end, :] += self.filters[:, :, :, f] * output_gradient[b, i, j, f]

                # Gradient w.r.t bias (sum over all positions)
                self.bias_gradient[f] = np.sum(output_gradient[:, :, :, f])

        # Remove padding from input gradient if it was added
        if self.padding == 'same' and pad_size > 0:
            input_gradient = padded_input_gradient[:, pad_size:-pad_size, pad_size:-pad_size, :]
        else:
            input_gradient = padded_input_gradient

        return input_gradient
