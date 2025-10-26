# Neural Networks from Scratch

A from-scratch implementation of neural networks and CNNs using only NumPy. Built for educational purposes to understand the fundamentals of deep learning without high-level frameworks.

## Features

- **Fully Connected Networks**: Dense layers with backpropagation
- **Convolutional Neural Networks**: Conv2D, MaxPooling, and modern CNN architectures
- **Optimizers**: Adam optimizer with momentum and adaptive learning rates
- **Activation Functions**: ReLU, Tanh, Sigmoid, Softmax
- **Loss Functions**: MSE, Binary Cross-Entropy, Categorical Cross-Entropy

## Architecture

The framework uses an object-oriented layer-based design where each component inherits from a base `Layer` class. All layers implement:
- `forward(input)`: Forward pass computation
- `backward(output_gradient)`: Backpropagation

## Project Structure

### Core Components

- `layer.py` - Base layer class
- `dense.py` / `dense_batch.py` - Fully connected layers
- `conv2d.py` - 2D convolutional layer
- `pooling.py` - Max/Min pooling layers
- `activation.py` - Activation functions (ReLU, Tanh, Sigmoid)
- `softmax.py` / `softmax_batch.py` - Softmax layer
- `reshape.py` - Flatten and reshape layers
- `adam.py` - Adam optimizer
- `losses.py` - Loss functions

### Models

- `vgg_model.py` - VGG-16 architecture for CIFAR-10
- `tiny_cnn_model.py` - Lightweight CNN (TinyNet, MicroNet)

### Training Scripts

- `simple_Neural_Network_XOR_test.py` - XOR problem test
- `simple_Neural_Network_Two_moons_test.py` - Two moons dataset test
- `tiny_cnn_train.py` - Train TinyNet on CIFAR-10
- `tiny_cnn_quick_test.py` - Quick test with reduced dataset

### Data Loaders

- `cifar_loader_numpy.py` - Pure NumPy CIFAR-10 loader (no TensorFlow dependency)
- `cifar_loader.py` - TensorFlow-based loader (alternative)

## Quick Start

### Train on XOR Problem

```bash
python simple_Neural_Network_XOR_test.py
```

### Train on Two Moons Dataset

```bash
python simple_Neural_Network_Two_moons_test.py
```

### Train CNN on CIFAR-10

#### Quick Test (1 hour)
```bash
python tiny_cnn_quick_test.py
```

#### Full Training
```bash
python tiny_cnn_train.py
```

## CNN Architectures

### TinyNet
- **Parameters**: 156,074
- **Architecture**: 3 conv blocks (16→32→64 filters) + 2 FC layers
- **CIFAR-10 Accuracy**: 60-70%
- **Training Time**: ~1 hour (quick test), 10-12 hours (full, CPU)

### MicroNet
- **Parameters**: 135,000
- **Architecture**: 2 conv blocks + 1 FC layer
- **Training Time**: 2x faster than TinyNet

### VGG-16 (Adapted)
- **Parameters**: 15M+
- **Architecture**: Full VGG-16 adapted for CIFAR-10
- **Training Time**: 10+ hours (CPU)

## GPU Acceleration

Install CuPy for automatic GPU acceleration:

```bash
pip install cupy-cuda11x  # Replace with your CUDA version
```

The code automatically detects and uses GPU if CuPy is available (10-20x speedup).

## Data Format

- **Images**: `(batch_size, height, width, channels)`
- **Labels**: `(batch_size, num_classes, 1)` (one-hot encoded)

## Requirements

- NumPy
- (Optional) CuPy for GPU acceleration
- (Optional) TensorFlow for CIFAR-10 loading

## Implementation Details

### Training Loop Pattern

```python
# Define network
network = [
    Conv2D(16, 3, 3, padding='same'),
    ReLU(),
    MaxPooling2D(2),
    Flatten(),
    DenseBatch(1024, 128),
    ReLU(),
    DenseBatch(128, 10),
    SoftmaxBatch()
]

# Create optimizer
optimizer = AdamCNN(learning_rate=0.001)

# Training
for epoch in range(epochs):
    for X_batch, y_batch in data:
        # Forward pass
        output = X_batch
        for layer in network:
            output = layer.forward(output)

        # Backward pass
        gradient = loss_prime(y_batch, output)
        for layer in reversed(network):
            gradient = layer.backward(gradient)
            if isinstance(layer, (Conv2D, DenseBatch)):
                optimizer.update(layer)
```

## Performance Notes

This is a **pure NumPy implementation** using nested loops for convolutions. It's designed for educational purposes to understand CNN internals, not for production use.

For production deep learning, use PyTorch or TensorFlow which have:
- Highly optimized C++/CUDA backends
- Vectorized operations
- Automatic differentiation
- Distributed training support

## Results

### XOR Problem
- Network: 2→3→1 with Tanh
- Accuracy: >95%

### Two Moons
- Network: 2→16→1 with ReLU
- Accuracy: >95%

### CIFAR-10 (TinyNet)
- Training samples: 45,000
- Test accuracy: 60-70%
- Comparison: Random guessing = 10%, State-of-art = 95%+

## License

MIT

## Author

Aditya Pramod Pawar
