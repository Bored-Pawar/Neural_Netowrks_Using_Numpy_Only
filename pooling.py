from layer import Layer
try:
    import cupy as np  # Use CuPy for GPU acceleration
except ImportError:
    import numpy as np  # Fallback to NumPy if CuPy not available

class MaxPooling2D(Layer):
    def __init__(self, pool_size=2, stride=None, padding='valid'):
        """
        Initialize a 2D Max Pooling Layer

        Args:
            pool_size: Size of the pooling window (e.g., 2 for 2x2)
            stride: Stride for pooling (default=pool_size)
            padding: 'valid' (no padding) or 'same' (zero padding)
        """
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.padding = padding
        self.max_indices = None  # Store indices of max values for backward pass

    def _apply_padding(self, input_data):
        """Apply zero padding to input if padding='same'"""
        if self.padding == 'valid':
            return input_data, 0

        # Calculate padding needed for 'same'
        pad_size = (self.pool_size - 1) // 2

        # Pad: (batch, height, width, channels)
        padded = np.pad(input_data,
                       ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                       mode='constant',
                       constant_values=-np.inf)  # Use -inf for max pooling padding
        return padded, pad_size

    def forward(self, input_data):
        """
        Forward pass of max pooling

        Args:
            input_data: Shape (batch_size, height, width, channels)

        Returns:
            output: Shape (batch_size, out_height, out_width, channels)
        """
        self.input = input_data
        batch_size, input_height, input_width, channels = input_data.shape

        # Apply padding if needed
        padded_input, pad_size = self._apply_padding(input_data)
        _, padded_height, padded_width, _ = padded_input.shape

        # Calculate output dimensions
        out_height = (padded_height - self.pool_size) // self.stride + 1
        out_width = (padded_width - self.pool_size) // self.stride + 1

        # Initialize output
        output = np.zeros((batch_size, out_height, out_width, channels))

        # Store max indices for backward pass
        # Store as (batch, out_h, out_w, channels, 2) where last dim is [h_idx, w_idx]
        self.max_indices = np.zeros((batch_size, out_height, out_width, channels, 2), dtype=int)

        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        # Extract the region
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        region = padded_input[b, h_start:h_end, w_start:w_end, c]

                        # Find max value and its position
                        max_val = np.max(region)
                        output[b, i, j, c] = max_val

                        # Store the position of max value (relative to region)
                        max_pos = np.unravel_index(np.argmax(region), region.shape)
                        self.max_indices[b, i, j, c, 0] = h_start + max_pos[0]
                        self.max_indices[b, i, j, c, 1] = w_start + max_pos[1]

        self.pad_size = pad_size  # Store for backward pass
        return output

    def backward(self, output_gradient):
        """
        Backward pass of max pooling

        Args:
            output_gradient: Gradient from next layer, shape (batch_size, out_height, out_width, channels)

        Returns:
            input_gradient: Gradient w.r.t input, shape (batch_size, height, width, channels)
        """
        batch_size, input_height, input_width, channels = self.input.shape
        _, out_height, out_width, _ = output_gradient.shape

        # Create gradient for padded input
        if self.padding == 'same' and self.pad_size > 0:
            padded_height = input_height + 2 * self.pad_size
            padded_width = input_width + 2 * self.pad_size
            padded_input_gradient = np.zeros((batch_size, padded_height, padded_width, channels))
        else:
            padded_input_gradient = np.zeros_like(self.input)

        # Route gradients to max positions
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        # Get the position where max was found
                        max_h = self.max_indices[b, i, j, c, 0]
                        max_w = self.max_indices[b, i, j, c, 1]

                        # Route gradient to that position
                        padded_input_gradient[b, max_h, max_w, c] += output_gradient[b, i, j, c]

        # Remove padding if it was added
        if self.padding == 'same' and self.pad_size > 0:
            input_gradient = padded_input_gradient[:, self.pad_size:-self.pad_size, self.pad_size:-self.pad_size, :]
        else:
            input_gradient = padded_input_gradient

        return input_gradient


class MinPooling2D(Layer):
    def __init__(self, pool_size=2, stride=None, padding='valid'):
        """
        Initialize a 2D Min Pooling Layer

        Args:
            pool_size: Size of the pooling window (e.g., 2 for 2x2)
            stride: Stride for pooling (default=pool_size)
            padding: 'valid' (no padding) or 'same' (zero padding)
        """
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.padding = padding
        self.min_indices = None  # Store indices of min values for backward pass

    def _apply_padding(self, input_data):
        """Apply zero padding to input if padding='same'"""
        if self.padding == 'valid':
            return input_data, 0

        # Calculate padding needed for 'same'
        pad_size = (self.pool_size - 1) // 2

        # Pad: (batch, height, width, channels)
        padded = np.pad(input_data,
                       ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                       mode='constant',
                       constant_values=np.inf)  # Use +inf for min pooling padding
        return padded, pad_size

    def forward(self, input_data):
        """
        Forward pass of min pooling

        Args:
            input_data: Shape (batch_size, height, width, channels)

        Returns:
            output: Shape (batch_size, out_height, out_width, channels)
        """
        self.input = input_data
        batch_size, input_height, input_width, channels = input_data.shape

        # Apply padding if needed
        padded_input, pad_size = self._apply_padding(input_data)
        _, padded_height, padded_width, _ = padded_input.shape

        # Calculate output dimensions
        out_height = (padded_height - self.pool_size) // self.stride + 1
        out_width = (padded_width - self.pool_size) // self.stride + 1

        # Initialize output
        output = np.zeros((batch_size, out_height, out_width, channels))

        # Store min indices for backward pass
        self.min_indices = np.zeros((batch_size, out_height, out_width, channels, 2), dtype=int)

        # Perform min pooling
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        # Extract the region
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        region = padded_input[b, h_start:h_end, w_start:w_end, c]

                        # Find min value and its position
                        min_val = np.min(region)
                        output[b, i, j, c] = min_val

                        # Store the position of min value (relative to region)
                        min_pos = np.unravel_index(np.argmin(region), region.shape)
                        self.min_indices[b, i, j, c, 0] = h_start + min_pos[0]
                        self.min_indices[b, i, j, c, 1] = w_start + min_pos[1]

        self.pad_size = pad_size  # Store for backward pass
        return output

    def backward(self, output_gradient):
        """
        Backward pass of min pooling

        Args:
            output_gradient: Gradient from next layer, shape (batch_size, out_height, out_width, channels)

        Returns:
            input_gradient: Gradient w.r.t input, shape (batch_size, height, width, channels)
        """
        batch_size, input_height, input_width, channels = self.input.shape
        _, out_height, out_width, _ = output_gradient.shape

        # Create gradient for padded input
        if self.padding == 'same' and self.pad_size > 0:
            padded_height = input_height + 2 * self.pad_size
            padded_width = input_width + 2 * self.pad_size
            padded_input_gradient = np.zeros((batch_size, padded_height, padded_width, channels))
        else:
            padded_input_gradient = np.zeros_like(self.input)

        # Route gradients to min positions
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        # Get the position where min was found
                        min_h = self.min_indices[b, i, j, c, 0]
                        min_w = self.min_indices[b, i, j, c, 1]

                        # Route gradient to that position
                        padded_input_gradient[b, min_h, min_w, c] += output_gradient[b, i, j, c]

        # Remove padding if it was added
        if self.padding == 'same' and self.pad_size > 0:
            input_gradient = padded_input_gradient[:, self.pad_size:-self.pad_size, self.pad_size:-self.pad_size, :]
        else:
            input_gradient = padded_input_gradient

        return input_gradient
