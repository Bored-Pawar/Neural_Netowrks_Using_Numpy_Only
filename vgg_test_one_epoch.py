"""
Quick Test Script: VGG-16 with 1 Epoch

Purpose: Fast test to verify GPU setup and code functionality
- Runs only 1 epoch instead of 10
- Uses simplified VGG-16
- Perfect for testing before full training run

Expected time: ~15-30 minutes on GPU
"""

try:
    import cupy as np  # Use CuPy for GPU acceleration
    GPU_AVAILABLE = True
    print("=" * 70)
    print("GPU ACCELERATION ENABLED (CuPy)")
    print("=" * 70)
except ImportError:
    import numpy as np  # Fallback to NumPy if CuPy not available
    GPU_AVAILABLE = False
    print("=" * 70)
    print("Running on CPU (NumPy). Install CuPy for GPU acceleration.")
    print("=" * 70)

from vgg_model import build_vgg16_cifar10, build_vgg16_simplified_cifar10
from cifar_loader import load_cifar10, split_train_val, get_cifar10_class_names
from adam import Adam, AdamCNN
from conv2d import Conv2D
from dense_batch import DenseBatch
from losses import CCE, CCE_prime
import time

def calculate_accuracy(predictions, labels):
    """
    Calculate accuracy given predictions and true labels

    Args:
        predictions: Shape (batch_size, num_classes, 1)
        labels: Shape (batch_size, num_classes, 1) - one-hot encoded

    Returns:
        accuracy: Float between 0 and 1
    """
    # Get predicted class (argmax along class dimension)
    pred_classes = np.argmax(predictions.squeeze(), axis=1)
    true_classes = np.argmax(labels.squeeze(), axis=1)

    accuracy = np.mean(pred_classes == true_classes)
    return accuracy


def evaluate_network(network, X, y, batch_size=100):
    """
    Evaluate network on a dataset

    Args:
        network: List of layers
        X: Images (n_samples, 32, 32, 3)
        y: Labels (n_samples, 10, 1)
        batch_size: Batch size for evaluation

    Returns:
        avg_loss: Average loss
        accuracy: Accuracy
    """
    n_samples = X.shape[0]
    total_loss = 0
    all_predictions = []
    all_labels = []

    # Process in batches
    for i in range(0, n_samples, batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]

        # Forward pass through network
        output = batch_X
        for layer in network:
            output = layer.forward(output)

        # Calculate loss (per sample in batch)
        # Output shape is (batch_size, num_classes, 1) after Softmax
        for j in range(batch_X.shape[0]):
            sample_pred = output[j:j+1, :, :]
            sample_label = batch_y[j:j+1, :, :]

            total_loss += CCE(sample_label, sample_pred)
            all_predictions.append(sample_pred)
            all_labels.append(sample_label)

    # Calculate metrics
    avg_loss = total_loss / n_samples
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    accuracy = calculate_accuracy(all_predictions, all_labels)

    return avg_loss, accuracy


def test_vgg_one_epoch(batch_size=32, learning_rate=0.001, val_size=5000):
    """
    Test VGG-16 with only 1 epoch - for quick verification

    Args:
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        val_size: Number of samples for validation set
    """
    print("=" * 70)
    print("VGG-16 ONE EPOCH TEST on CIFAR-10")
    print("=" * 70)

    # Load CIFAR-10 data
    print("\nLoading CIFAR-10 dataset...")
    X_train, y_train, X_test, y_test = load_cifar10(normalize=True, one_hot=True)

    # Split into train/val
    X_train, y_train, X_val, y_val = split_train_val(X_train, y_train, val_size=val_size)

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # Build network
    print(f"\nBuilding simplified VGG-16 network...")
    network = build_vgg16_simplified_cifar10()

    print(f"Network has {len(network)} layers")

    # Create optimizers
    optimizer_dense = Adam(learning_rate=learning_rate)
    optimizer_conv = AdamCNN(learning_rate=learning_rate)

    print(f"\nTest Configuration:")
    print(f"  Epochs: 1 (TESTING ONLY)")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batches per epoch: {X_train.shape[0] // batch_size}")

    # Training loop - ONLY 1 EPOCH
    print("\n" + "=" * 70)
    print("Starting Training (1 EPOCH TEST)...")
    print("=" * 70)

    epoch_start_time = time.time()

    # Shuffle training data
    indices = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    epoch_loss = 0
    n_batches = X_train.shape[0] // batch_size

    # Mini-batch training
    for batch_idx in range(n_batches):
        # Get batch
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_X = X_train_shuffled[start_idx:end_idx]
        batch_y = y_train_shuffled[start_idx:end_idx]

        # Forward pass
        output = batch_X
        for layer in network:
            output = layer.forward(output)

        # Calculate loss for each sample in batch
        batch_loss = 0
        for i in range(batch_size):
            # Output shape is (batch_size, num_classes, 1) after Softmax
            sample_pred = output[i:i+1, :, :]
            sample_label = batch_y[i:i+1, :, :]

            batch_loss += CCE(sample_label, sample_pred)

        epoch_loss += batch_loss

        # Backward pass - compute gradient for entire batch
        gradient = np.zeros_like(output)

        for i in range(batch_size):
            sample_pred = output[i:i+1, :, :]
            sample_label = batch_y[i:i+1, :, :]

            # Compute gradient for this sample
            sample_gradient = CCE_prime(sample_label, sample_pred)

            gradient[i:i+1, :, :] = sample_gradient

        # Backpropagate through network
        for layer in reversed(network):
            gradient = layer.backward(gradient)

            # Update weights
            if isinstance(layer, Conv2D):
                optimizer_conv.update(layer)
            elif isinstance(layer, DenseBatch):
                optimizer_dense.update(layer)

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            avg_batch_loss = batch_loss / batch_size
            elapsed = time.time() - epoch_start_time
            batches_per_sec = (batch_idx + 1) / elapsed
            eta_seconds = (n_batches - batch_idx - 1) / batches_per_sec
            eta_minutes = eta_seconds / 60

            print(f"  Batch {batch_idx+1}/{n_batches} - Loss: {avg_batch_loss:.4f} - "
                  f"Speed: {batches_per_sec:.2f} batch/s - ETA: {eta_minutes:.1f} min")

    # Calculate epoch metrics
    avg_epoch_loss = epoch_loss / (n_batches * batch_size)
    epoch_time = time.time() - epoch_start_time

    # Evaluate on validation set
    print(f"\n  Evaluating on validation set...")
    val_loss, val_acc = evaluate_network(network, X_val, y_val, batch_size=100)

    print(f"\n" + "=" * 70)
    print(f"EPOCH 1 COMPLETE!")
    print("=" * 70)
    print(f"  Train Loss: {avg_epoch_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val Accuracy: {val_acc*100:.2f}%")
    print(f"  Time: {epoch_time/60:.2f} minutes ({epoch_time:.1f} seconds)")
    print("=" * 70)

    # Estimate time for full training
    print(f"\nEstimated time for 10 epochs: {(epoch_time * 10) / 3600:.2f} hours")

    # Optional: Quick test set evaluation
    print("\n" + "=" * 70)
    print("Quick Test Set Evaluation...")
    print("=" * 70)

    test_loss, test_acc = evaluate_network(network, X_test, y_test, batch_size=100)

    print(f"\nTest Results (after 1 epoch only):")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"\nNote: Accuracy will improve significantly with more epochs!")

    print("\nClass names:", get_cifar10_class_names())

    print("\n" + "=" * 70)
    print("TEST COMPLETE! GPU is working correctly.")
    print("To run full training, use: python vgg_cifar_test.py")
    print("=" * 70)


if __name__ == "__main__":
    # Run 1 epoch test
    test_vgg_one_epoch(
        batch_size=32,
        learning_rate=0.001,
        val_size=5000
    )
