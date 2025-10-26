from conv2d import Conv2D
from pooling import MaxPooling2D
from reshape import Flatten
from dense_batch import DenseBatch  # Use batch-processing Dense layer for CNN
from activation import ReLU
from softmax_batch import SoftmaxBatch  # Use batch-processing Softmax for CNN

def build_vgg16_cifar10():
    """
    Build VGG-16 architecture adapted for CIFAR-10

    CIFAR-10 images are 32x32x3, much smaller than ImageNet's 224x224x3
    This is a simplified VGG-16 with adjusted architecture for smaller input size

    Architecture:
    - Input: 32x32x3
    - Block 1: Conv64 -> Conv64 -> MaxPool -> 16x16x64
    - Block 2: Conv128 -> Conv128 -> MaxPool -> 8x8x128
    - Block 3: Conv256 -> Conv256 -> Conv256 -> MaxPool -> 4x4x256
    - Block 4: Conv512 -> Conv512 -> Conv512 -> MaxPool -> 2x2x512
    - Block 5: Conv512 -> Conv512 -> Conv512 -> MaxPool -> 1x1x512
    - Flatten -> 512
    - FC 512 -> ReLU
    - FC 512 -> ReLU
    - FC 10 (output classes)
    - Softmax

    Returns:
        network: List of layers
    """

    network = [
        # Block 1: 32x32x3 -> 32x32x64 -> 16x16x64
        Conv2D(num_filters=64, kernel_size=3, input_channels=3, stride=1, padding='same'),
        ReLU(),
        Conv2D(num_filters=64, kernel_size=3, input_channels=64, stride=1, padding='same'),
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2, padding='valid'),  # 32x32 -> 16x16

        # Block 2: 16x16x64 -> 16x16x128 -> 8x8x128
        Conv2D(num_filters=128, kernel_size=3, input_channels=64, stride=1, padding='same'),
        ReLU(),
        Conv2D(num_filters=128, kernel_size=3, input_channels=128, stride=1, padding='same'),
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2, padding='valid'),  # 16x16 -> 8x8

        # Block 3: 8x8x128 -> 8x8x256 -> 4x4x256
        Conv2D(num_filters=256, kernel_size=3, input_channels=128, stride=1, padding='same'),
        ReLU(),
        Conv2D(num_filters=256, kernel_size=3, input_channels=256, stride=1, padding='same'),
        ReLU(),
        Conv2D(num_filters=256, kernel_size=3, input_channels=256, stride=1, padding='same'),
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2, padding='valid'),  # 8x8 -> 4x4

        # Block 4: 4x4x256 -> 4x4x512 -> 2x2x512
        Conv2D(num_filters=512, kernel_size=3, input_channels=256, stride=1, padding='same'),
        ReLU(),
        Conv2D(num_filters=512, kernel_size=3, input_channels=512, stride=1, padding='same'),
        ReLU(),
        Conv2D(num_filters=512, kernel_size=3, input_channels=512, stride=1, padding='same'),
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2, padding='valid'),  # 4x4 -> 2x2

        # Block 5: 2x2x512 -> 2x2x512 -> 1x1x512
        Conv2D(num_filters=512, kernel_size=3, input_channels=512, stride=1, padding='same'),
        ReLU(),
        Conv2D(num_filters=512, kernel_size=3, input_channels=512, stride=1, padding='same'),
        ReLU(),
        Conv2D(num_filters=512, kernel_size=3, input_channels=512, stride=1, padding='same'),
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2, padding='valid'),  # 2x2 -> 1x1

        # Flatten: 1x1x512 -> 512
        Flatten(),  # Output: (batch, 512, 1)

        # Fully Connected Layers
        DenseBatch(512, 512),  # FC1
        ReLU(),
        DenseBatch(512, 512),  # FC2
        ReLU(),
        DenseBatch(512, 10),   # FC3 - Output layer (10 classes for CIFAR-10)

        # Softmax for classification
        SoftmaxBatch()
    ]

    return network


def build_vgg16_simplified_cifar10():
    """
    Build a simplified VGG-16 for CIFAR-10 with fewer filters
    This version trains faster and uses less memory while maintaining the architecture

    Architecture uses half the filters in each layer compared to original VGG-16

    Returns:
        network: List of layers
    """

    network = [
        # Block 1: 32x32x3 -> 32x32x32 -> 16x16x32
        Conv2D(num_filters=32, kernel_size=3, input_channels=3, stride=1, padding='same'),
        ReLU(),
        Conv2D(num_filters=32, kernel_size=3, input_channels=32, stride=1, padding='same'),
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2, padding='valid'),

        # Block 2: 16x16x32 -> 16x16x64 -> 8x8x64
        Conv2D(num_filters=64, kernel_size=3, input_channels=32, stride=1, padding='same'),
        ReLU(),
        Conv2D(num_filters=64, kernel_size=3, input_channels=64, stride=1, padding='same'),
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2, padding='valid'),

        # Block 3: 8x8x64 -> 8x8x128 -> 4x4x128
        Conv2D(num_filters=128, kernel_size=3, input_channels=64, stride=1, padding='same'),
        ReLU(),
        Conv2D(num_filters=128, kernel_size=3, input_channels=128, stride=1, padding='same'),
        ReLU(),
        Conv2D(num_filters=128, kernel_size=3, input_channels=128, stride=1, padding='same'),
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2, padding='valid'),

        # Block 4: 4x4x128 -> 4x4x256 -> 2x2x256
        Conv2D(num_filters=256, kernel_size=3, input_channels=128, stride=1, padding='same'),
        ReLU(),
        Conv2D(num_filters=256, kernel_size=3, input_channels=256, stride=1, padding='same'),
        ReLU(),
        Conv2D(num_filters=256, kernel_size=3, input_channels=256, stride=1, padding='same'),
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2, padding='valid'),

        # Block 5: 2x2x256 -> 2x2x256 -> 1x1x256
        Conv2D(num_filters=256, kernel_size=3, input_channels=256, stride=1, padding='same'),
        ReLU(),
        Conv2D(num_filters=256, kernel_size=3, input_channels=256, stride=1, padding='same'),
        ReLU(),
        Conv2D(num_filters=256, kernel_size=3, input_channels=256, stride=1, padding='same'),
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2, padding='valid'),

        # Flatten: 1x1x256 -> 256
        Flatten(),

        # Fully Connected Layers
        DenseBatch(256, 256),
        ReLU(),
        DenseBatch(256, 256),
        ReLU(),
        DenseBatch(256, 10),

        # Softmax for classification
        SoftmaxBatch()
    ]

    return network
