"""
Pure NumPy CIFAR-10 loader (no TensorFlow dependency)
Downloads CIFAR-10 directly from the official source
"""

import numpy as np
import os
import pickle
import urllib.request
import tarfile

def download_cifar10(data_dir='./cifar-10-data'):
    """
    Download CIFAR-10 dataset from official source

    Args:
        data_dir: Directory to save the dataset

    Returns:
        Path to extracted data directory
    """
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = 'cifar-10-python.tar.gz'
    filepath = os.path.join(data_dir, filename)

    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Check if already downloaded
    extract_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    if os.path.exists(extract_dir):
        print(f"CIFAR-10 already exists at {extract_dir}")
        return extract_dir

    # Download
    print(f"Downloading CIFAR-10 from {url}...")
    print("This may take a few minutes...")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rDownloading: {percent}%", end='')

    urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
    print("\nDownload complete!")

    # Extract
    print(f"Extracting {filename}...")
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(data_dir)
    print("Extraction complete!")

    return extract_dir


def load_cifar10_batch(file_path):
    """
    Load a single CIFAR-10 batch file

    Args:
        file_path: Path to the batch file

    Returns:
        data: Images (10000, 32, 32, 3)
        labels: Labels (10000,)
    """
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')

    # Extract data and labels
    data = batch[b'data']
    labels = batch[b'labels']

    # Reshape data from (10000, 3072) to (10000, 32, 32, 3)
    # CIFAR-10 stores as [red_channel, green_channel, blue_channel] each 1024 values
    data = data.reshape(-1, 3, 32, 32)  # (10000, 3, 32, 32)
    data = data.transpose(0, 2, 3, 1)    # (10000, 32, 32, 3)

    return data, np.array(labels)


def load_cifar10(normalize=True, one_hot=True, data_dir='./cifar-10-data'):
    """
    Load the complete CIFAR-10 dataset using pure NumPy

    Args:
        normalize: If True, normalize pixel values to [0, 1]
        one_hot: If True, convert labels to one-hot encoding
        data_dir: Directory to save/load the dataset

    Returns:
        X_train: Training images (50000, 32, 32, 3)
        y_train: Training labels (50000, 10, 1) if one_hot, else (50000,)
        X_test: Test images (10000, 32, 32, 3)
        y_test: Test labels (10000, 10, 1) if one_hot, else (10000,)
    """
    print("Loading CIFAR-10 dataset (pure NumPy)...")

    # Download if necessary
    extract_dir = download_cifar10(data_dir)

    # Load training batches
    X_train_list = []
    y_train_list = []

    for i in range(1, 6):
        batch_file = os.path.join(extract_dir, f'data_batch_{i}')
        print(f"Loading training batch {i}/5...")
        data, labels = load_cifar10_batch(batch_file)
        X_train_list.append(data)
        y_train_list.append(labels)

    # Concatenate all training batches
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    # Load test batch
    print("Loading test batch...")
    test_file = os.path.join(extract_dir, 'test_batch')
    X_test, y_test = load_cifar10_batch(test_file)

    print(f"Loaded: Train={X_train.shape}, Test={X_test.shape}")

    # Normalize to [0, 1]
    if normalize:
        print("Normalizing to [0, 1]...")
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

    # Convert labels to one-hot encoding
    if one_hot:
        print("Converting labels to one-hot encoding...")
        y_train = to_one_hot(y_train, num_classes=10)
        y_test = to_one_hot(y_test, num_classes=10)

    print("CIFAR-10 loaded successfully!")
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test: X={X_test.shape}, y={y_test.shape}")

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
    print("Testing pure NumPy CIFAR-10 loader...")

    X_train, y_train, X_test, y_test = load_cifar10()

    print(f"\nTraining set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    print(f"Image range: [{X_train.min()}, {X_train.max()}]")

    # Split into train/val
    X_train_split, y_train_split, X_val, y_val = split_train_val(X_train, y_train)

    print(f"\nAfter train/val split:")
    print(f"Training set: {X_train_split.shape}, {y_train_split.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")

    print("\nClass names:", get_cifar10_class_names())

    print("\n✅ All tests passed!")
