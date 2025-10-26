"""
QUICK TEST version of TinyNet - Trains in under 1 hour on CPU
Uses reduced dataset and epochs for demonstration
"""

import time
import numpy as np
from tiny_cnn_model import build_tiny_cnn, build_micro_cnn
from cifar_loader_numpy import load_cifar10, split_train_val
from losses import CCE, CCE_prime
from adam import Adam, AdamCNN
from conv2d import Conv2D
from dense_batch import DenseBatch

print("=" * 70)
print("TINYNET QUICK TEST - Optimized for 1 Hour Training")
print("=" * 70)
print("\nNOTE: This is a REDUCED VERSION for quick testing")
print("Full training would take many hours on CPU\n")

# Load data
print("Loading CIFAR-10...")
X_train, y_train, X_test, y_test = load_cifar10(normalize=True, one_hot=True)
X_train, y_train, X_val, y_val = split_train_val(X_train, y_train, val_size=1000)

# REDUCE DATASET for quick testing
print("\nUsing REDUCED dataset for quick testing:")
X_train = X_train[:1000]  # Only 1000 training samples
y_train = y_train[:1000]
X_val = X_val[:500]       # Only 500 validation samples
y_val = y_val[:500]

print(f"  Train: {X_train.shape[0]} samples")
print(f"  Val: {X_val.shape[0]} samples")
print(f"  Test: {X_test.shape[0]} samples")

# Build network
print("\nBuilding TinyNet...")
network = build_tiny_cnn()

# REDUCED CONFIGURATION
EPOCHS = 3              # Only 3 epochs
BATCH_SIZE = 32         # Smaller batches
LEARNING_RATE = 0.001

n_batches = X_train.shape[0] // BATCH_SIZE

print(f"\nConfiguration:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Batches per epoch: {n_batches}")
print(f"  Total batches: {EPOCHS * n_batches}")

# Estimate time
print(f"\nEstimated time (at ~1.5 min/batch): {EPOCHS * n_batches * 1.5 / 60:.1f} hours")
print("=" * 70)

# Create optimizers
optimizer_dense = Adam(learning_rate=LEARNING_RATE)
optimizer_conv = AdamCNN(learning_rate=LEARNING_RATE)

# Helper functions
def predict_batch(network, X_batch):
    output = X_batch
    for layer in network:
        output = layer.forward(output)
    return output

def calculate_accuracy(network, X, y):
    predictions = predict_batch(network, X)
    pred_classes = np.argmax(predictions.squeeze(), axis=1)
    true_classes = np.argmax(y.squeeze(), axis=1)
    return float(np.sum(pred_classes == true_classes)) / len(y) * 100

# Training
print("\nStarting training...\n")
training_start = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    epoch_loss = 0

    # Shuffle
    indices = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    # Mini-batch training
    for batch_idx in range(n_batches):
        batch_start = time.time()

        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        X_batch = X_train_shuffled[start_idx:end_idx]
        y_batch = y_train_shuffled[start_idx:end_idx]

        # Forward
        output = predict_batch(network, X_batch)

        # Loss
        batch_loss = sum(CCE(y_batch[i], output[i]) for i in range(BATCH_SIZE)) / BATCH_SIZE
        epoch_loss += batch_loss

        # Backward
        gradient = CCE_prime(y_batch, output)
        for layer in reversed(network):
            gradient = layer.backward(gradient)
            if isinstance(layer, Conv2D):
                optimizer_conv.update(layer)
            elif isinstance(layer, DenseBatch):
                optimizer_dense.update(layer)

        batch_time = time.time() - batch_start

        # Print progress
        if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
            remaining_batches = (EPOCHS - epoch) * n_batches - batch_idx - 1
            eta_minutes = remaining_batches * batch_time / 60
            print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{n_batches} | "
                  f"Loss: {batch_loss:.4f} | Time: {batch_time:.1f}s | ETA: {eta_minutes:.1f}m")

    # Epoch summary
    epoch_time = time.time() - epoch_start
    avg_loss = epoch_loss / n_batches

    print(f"\n{'='*70}")
    print(f"EPOCH {epoch+1}/{EPOCHS} COMPLETE")
    print(f"{'='*70}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Time: {epoch_time/60:.2f} minutes")

    # Calculate accuracies
    print("Evaluating...")
    train_acc = calculate_accuracy(network, X_train[:100], y_train[:100])
    val_acc = calculate_accuracy(network, X_val, y_val)

    print(f"Train Accuracy (100 samples): {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")

    elapsed = (time.time() - training_start) / 60
    print(f"Elapsed: {elapsed:.2f} minutes")
    print(f"{'='*70}\n")

# Final test
print("\nFinal Evaluation...")
test_acc = calculate_accuracy(network, X_test[:1000], y_test[:1000])

total_time = (time.time() - training_start) / 60

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Total Time: {total_time:.2f} minutes ({total_time/60:.2f} hours)")
print(f"Test Accuracy (1000 samples): {test_acc:.2f}%")
print("=" * 70)

# Save results
with open("tiny_cnn_quick_results.txt", "w") as f:
    f.write("TinyNet Quick Test Results\n")
    f.write("=" * 50 + "\n")
    f.write(f"Training samples: {X_train.shape[0]}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Training time: {total_time:.2f} minutes\n")
    f.write(f"Test accuracy: {test_acc:.2f}%\n")
    f.write("\nNOTE: This was a quick test with reduced dataset.\n")
    f.write("Full training requires GPU or much longer time.\n")

print("\nResults saved to tiny_cnn_quick_results.txt")
