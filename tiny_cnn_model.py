from conv2d import Conv2D
from pooling import MaxPooling2D
from reshape import Flatten
from dense_batch import DenseBatch
from activation import ReLU
from softmax_batch import SoftmaxBatch

def build_tiny_cnn():
    """
    Build a very small and efficient CNN for CIFAR-10
    Designed to train within 1 hour while achieving good accuracy

    Architecture (TinyNet):
    - Input: 32x32x3
    - Block 1: Conv16 3x3 -> ReLU -> MaxPool 2x2 -> 16x16x16
    - Block 2: Conv32 3x3 -> ReLU -> MaxPool 2x2 -> 8x8x32
    - Block 3: Conv64 3x3 -> ReLU -> MaxPool 2x2 -> 4x4x64
    - Flatten -> 1024
    - FC 128 -> ReLU
    - FC 10 (output classes)
    - Softmax

    Total parameters: ~140K (vs VGG's 15M+)
    Expected accuracy: 60-70% on CIFAR-10

    Returns:
        network: List of layers
    """

    network = [
        # Block 1: 32x32x3 -> 32x32x16 -> 16x16x16
        Conv2D(num_filters=16, kernel_size=3, input_channels=3, stride=1, padding='same'),
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2, padding='valid'),

        # Block 2: 16x16x16 -> 16x16x32 -> 8x8x32
        Conv2D(num_filters=32, kernel_size=3, input_channels=16, stride=1, padding='same'),
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2, padding='valid'),

        # Block 3: 8x8x32 -> 8x8x64 -> 4x4x64
        Conv2D(num_filters=64, kernel_size=3, input_channels=32, stride=1, padding='same'),
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2, padding='valid'),

        # Flatten: 4x4x64 -> 1024
        Flatten(),

        # Fully Connected Layers
        DenseBatch(1024, 128),  # Single hidden layer
        ReLU(),
        DenseBatch(128, 10),    # Output layer (10 classes)

        # Softmax for classification
        SoftmaxBatch()
    ]

    return network


def build_micro_cnn():
    """
    Build an even smaller CNN for CIFAR-10 (if TinyNet is still too slow)
    Extremely lightweight for faster training

    Architecture (MicroNet):
    - Input: 32x32x3
    - Conv16 3x3 -> ReLU -> MaxPool 2x2 -> 16x16x16
    - Conv32 3x3 -> ReLU -> MaxPool 2x2 -> 8x8x32
    - Flatten -> 2048
    - FC 64 -> ReLU
    - FC 10
    - Softmax

    Total parameters: ~135K
    Expected accuracy: 55-65% on CIFAR-10
    Trains ~2x faster than TinyNet

    Returns:
        network: List of layers
    """

    network = [
        # Block 1: 32x32x3 -> 32x32x16 -> 16x16x16
        Conv2D(num_filters=16, kernel_size=3, input_channels=3, stride=1, padding='same'),
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2, padding='valid'),

        # Block 2: 16x16x16 -> 16x16x32 -> 8x8x32
        Conv2D(num_filters=32, kernel_size=3, input_channels=16, stride=1, padding='same'),
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2, padding='valid'),

        # Flatten: 8x8x32 -> 2048
        Flatten(),

        # Fully Connected Layers
        DenseBatch(2048, 64),   # Small hidden layer
        ReLU(),
        DenseBatch(64, 10),     # Output layer

        # Softmax
        SoftmaxBatch()
    ]

    return network


if __name__ == "__main__":
    print("Building TinyNet CNN...")
    network = build_tiny_cnn()

    print("\nNetwork Architecture:")
    print("=" * 60)
    for i, layer in enumerate(network):
        print(f"Layer {i+1}: {layer.__class__.__name__}")
    print("=" * 60)

    # Count parameters
    total_params = 0
    for layer in network:
        if hasattr(layer, 'filters'):  # Conv2D layer
            params = layer.filters.size + layer.bias.size
            print(f"{layer.__class__.__name__}: {params:,} parameters")
            total_params += params
        elif hasattr(layer, 'weights'):  # Dense layer
            params = layer.weights.size + layer.bias.size
            print(f"{layer.__class__.__name__}: {params:,} parameters")
            total_params += params

    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"Estimated memory: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")
