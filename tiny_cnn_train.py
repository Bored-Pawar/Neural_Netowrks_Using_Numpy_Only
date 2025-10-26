"""
Efficient training script for TinyNet CNN on CIFAR-10
Designed to train within 1 hour maximum
"""

import time
import numpy as np
from tiny_cnn_model import build_tiny_cnn, build_micro_cnn
from cifar_loader_numpy import load_cifar10, split_train_val  # Pure NumPy loader (no TensorFlow!)
from losses import CCE, CCE_prime
from adam import Adam, AdamCNN
from conv2d import Conv2D
from dense_batch import DenseBatch

try:
    import cupy as cp
    GPU_AVAILABLE = True
    xp = cp
    print("GPU (CuPy) detected! Training will use GPU acceleration.")
except ImportError:
    xp = np
    GPU_AVAILABLE = False
    print("CuPy not found. Training on CPU (will be slower).")


def predict_batch(network, X_batch):
    """
    Forward pass through the network for a batch

    Args:
        network: List of layers
        X_batch: Input batch (batch_size, 32, 32, 3)

    Returns:
        output: Network predictions (batch_size, 10, 1)
    """
    output = X_batch
    for layer in network:
        output = layer.forward(output)
    return output


def calculate_accuracy(network, X, y, batch_size=100):
    """
    Calculate accuracy on a dataset

    Args:
        network: List of layers
        X: Images (n_samples, 32, 32, 3)
        y: One-hot labels (n_samples, 10, 1)
        batch_size: Batch size for evaluation

    Returns:
        accuracy: Accuracy percentage
    """
    n_samples = X.shape[0]
    correct = 0

    for i in range(0, n_samples, batch_size):
        # Get batch
        end_idx = min(i + batch_size, n_samples)
        X_batch = X[i:end_idx]
        y_batch = y[i:end_idx]

        # Predict
        predictions = predict_batch(network, X_batch)

        # Convert to class indices
        pred_classes = xp.argmax(predictions.squeeze(), axis=1)
        true_classes = xp.argmax(y_batch.squeeze(), axis=1)

        # Count correct predictions
        correct += xp.sum(pred_classes == true_classes)

    accuracy = float(correct) / n_samples * 100
    return accuracy


def train_cnn(network, X_train, y_train, X_val, y_val,
              epochs=10, batch_size=32, learning_rate=0.001,
              print_every=100):
    """
    Train the CNN with mini-batch gradient descent

    Args:
        network: List of layers
        X_train: Training images (n_samples, 32, 32, 3)
        y_train: Training labels (n_samples, 10, 1)
        X_val: Validation images
        y_val: Validation labels
        epochs: Number of epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        print_every: Print progress every N batches
    """

    # Create optimizers
    optimizer_dense = Adam(learning_rate=learning_rate)
    optimizer_conv = AdamCNN(learning_rate=learning_rate)

    n_train = X_train.shape[0]
    n_batches = n_train // batch_size

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Training samples: {n_train:,}")
    print(f"Validation samples: {X_val.shape[0]:,}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {n_batches}")
    print(f"Learning rate: {learning_rate}")
    print(f"GPU acceleration: {'Yes (CuPy)' if GPU_AVAILABLE else 'No (NumPy/CPU)'}")
    print("=" * 70)

    # Track training progress
    training_start = time.time()
    epoch_times = []

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0
        batch_times = []

        # Shuffle training data
        indices = xp.random.permutation(n_train)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        # Mini-batch training
        for batch_idx in range(n_batches):
            batch_start = time.time()

            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            # Forward pass
            output = predict_batch(network, X_batch)

            # Calculate loss (average over batch)
            batch_loss = 0
            for i in range(batch_size):
                batch_loss += CCE(y_batch[i], output[i])
            batch_loss /= batch_size
            epoch_loss += batch_loss

            # Backward pass
            gradient = CCE_prime(y_batch, output)

            for layer in reversed(network):
                gradient = layer.backward(gradient)

                # Update layer parameters
                if isinstance(layer, Conv2D):
                    optimizer_conv.update(layer)
                elif isinstance(layer, DenseBatch):
                    optimizer_dense.update(layer)

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Print progress
            if (batch_idx + 1) % print_every == 0 or batch_idx == 0:
                avg_batch_time = sum(batch_times[-print_every:]) / len(batch_times[-print_every:])
                batches_remaining = n_batches - batch_idx - 1
                eta_seconds = batches_remaining * avg_batch_time
                eta_minutes = eta_seconds / 60

                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{n_batches} | "
                      f"Loss: {batch_loss:.4f} | "
                      f"Time/batch: {avg_batch_time:.2f}s | "
                      f"ETA: {eta_minutes:.1f}m")

        # Epoch statistics
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_loss = epoch_loss / n_batches

        # Calculate accuracies
        print("\nEvaluating accuracy...")
        train_acc = calculate_accuracy(network, X_train[:1000], y_train[:1000], batch_size=100)
        val_acc = calculate_accuracy(network, X_val, y_val, batch_size=100)

        # Print epoch summary
        print("\n" + "=" * 70)
        print(f"EPOCH {epoch+1}/{epochs} SUMMARY")
        print("=" * 70)
        print(f"Average Loss: {avg_epoch_loss:.4f}")
        print(f"Train Accuracy (1000 samples): {train_acc:.2f}%")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        print(f"Epoch Time: {epoch_time/60:.2f} minutes")

        # Time estimates
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - epoch - 1
        eta_total = remaining_epochs * avg_epoch_time / 60
        elapsed_total = (time.time() - training_start) / 60

        print(f"Elapsed Time: {elapsed_total:.2f} minutes")
        print(f"Estimated Time Remaining: {eta_total:.2f} minutes")
        print(f"Estimated Total Time: {elapsed_total + eta_total:.2f} minutes")
        print("=" * 70 + "\n")

    # Final statistics
    total_time = (time.time() - training_start) / 60
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total Training Time: {total_time:.2f} minutes ({total_time/60:.2f} hours)")
    print(f"Final Validation Accuracy: {val_acc:.2f}%")
    print("=" * 70)


def main():
    """Main training function"""

    print("Loading CIFAR-10 dataset...")
    X_train, y_train, X_test, y_test = load_cifar10(normalize=True, one_hot=True)

    # Split into train and validation
    X_train, y_train, X_val, y_val = split_train_val(X_train, y_train, val_size=5000)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Build network
    print("\nBuilding TinyNet CNN...")
    network = build_tiny_cnn()

    print("\nNetwork layers:")
    for i, layer in enumerate(network):
        print(f"  {i+1}. {layer.__class__.__name__}")

    # Training configuration for 1 hour limit
    # Adjust these if training is too slow:
    # - Reduce epochs
    # - Reduce training samples (use X_train[:10000])
    # - Use build_micro_cnn() instead

    EPOCHS = 15              # Number of epochs (adjust based on speed)
    BATCH_SIZE = 64          # Batch size (larger = faster but more memory)
    LEARNING_RATE = 0.001    # Learning rate
    PRINT_EVERY = 50         # Print progress frequency

    # Optional: Train on subset for faster testing
    # X_train = X_train[:10000]
    # y_train = y_train[:10000]

    # Train
    train_cnn(
        network, X_train, y_train, X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        print_every=PRINT_EVERY
    )

    # Final test accuracy
    print("\nCalculating final test accuracy...")
    test_acc = calculate_accuracy(network, X_test, y_test, batch_size=100)
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    # Save results
    with open("tiny_cnn_results.txt", "w") as f:
        f.write("TinyNet CNN - CIFAR-10 Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Final Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Training samples: {X_train.shape[0]}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"GPU: {GPU_AVAILABLE}\n")

    print("\nResults saved to tiny_cnn_results.txt")


if __name__ == "__main__":
    main()
