import numpy as np
import tensorflow as tf

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

def load_cifar10(normalize=True, one_hot=True):
    """
    Load the complete CIFAR-10 dataset using TensorFlow/Keras

    Args:
        normalize: If True, normalize pixel values to [0, 1]
        one_hot: If True, convert labels to one-hot encoding

    Returns:
        X_train: Training images (50000, 32, 32, 3)
        y_train: Training labels (50000, 10, 1) if one_hot, else (50000,)
        X_test: Test images (10000, 32, 32, 3)
        y_test: Test labels (10000, 10, 1) if one_hot, else (10000,)
    """
    print("Loading CIFAR-10 dataset using TensorFlow...")

    # Load CIFAR-10 using Keras (automatically downloads and caches)
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Remove extra dimension from labels: (50000, 1) -> (50000,)
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    # Normalize to [0, 1]
    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

    # Convert labels to one-hot encoding
    if one_hot:
        y_train = to_one_hot(y_train, num_classes=10)
        y_test = to_one_hot(y_test, num_classes=10)

    print("CIFAR-10 loaded successfully!")

    # Convert to CuPy arrays if GPU is available
    if GPU_AVAILABLE:
        print("Converting data to GPU (CuPy arrays)...")
        X_train = cp.array(X_train)
        y_train = cp.array(y_train)
        X_test = cp.array(X_test)
        y_test = cp.array(y_test)
        print("Data moved to GPU!")

    return X_train, y_train, X_test, y_test


def to_one_hot(labels, num_classes=10):
    """
    Convert labels to one-hot encoding

    Args:
        labels: numpy array of shape (n,)
        num_classes: Number of classes

    Returns:
        one_hot: numpy array of shape (n, num_classes, 1)
    """
    n = labels.shape[0]
    one_hot = np.zeros((n, num_classes, 1))
    one_hot[np.arange(n), labels, 0] = 1
    return one_hot


def split_train_val(X_train, y_train, val_size=5000, seed=42):
    """
    Split training data into train and validation sets

    Args:
        X_train: Training images
        y_train: Training labels
        val_size: Number of samples for validation
        seed: Random seed for reproducibility

    Returns:
        X_train_split: Training images after split
        y_train_split: Training labels after split
        X_val: Validation images
        y_val: Validation labels
    """
    np.random.seed(seed)

    # Generate random indices
    n_samples = X_train.shape[0]
    indices = np.random.permutation(n_samples)

    # Split indices
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # Split data
    X_train_split = X_train[train_indices]
    y_train_split = y_train[train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]

    return X_train_split, y_train_split, X_val, y_val


def get_cifar10_class_names():
    """
    Get CIFAR-10 class names

    Returns:
        List of class names
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']


if __name__ == "__main__":
    # Test the loader
    print("Testing CIFAR-10 loader...")

    X_train, y_train, X_test, y_test = load_cifar10()

    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    print(f"Image range: [{X_train.min()}, {X_train.max()}]")

    # Split into train/val
    X_train_split, y_train_split, X_val, y_val = split_train_val(X_train, y_train)

    print(f"\nAfter train/val split:")
    print(f"Training set: {X_train_split.shape}, {y_train_split.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")

    print("\nClass names:", get_cifar10_class_names())
