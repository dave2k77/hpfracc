#!/usr/bin/env python3
"""
Machine Learning Integration Demo

This script demonstrates the comprehensive ML integration system for hpfracc,
including fractional neural networks, layers, attention mechanisms, and optimizers.

✅ Features Demonstrated:
- Fractional Neural Networks (FNN)
- Fractional Layers (Conv1D, Conv2D, LSTM, Transformer)
- Fractional Attention Mechanisms
- Fractional Loss Functions
- Fractional Optimizers
"""

import sys
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import hpfracc ML components
from hpfracc.ml import (
    FractionalNeuralNetwork,
    FractionalAttention,
    FractionalMSELoss,
    FractionalAdam,
    FractionalConv1D,
    FractionalConv2D,
    FractionalLSTM,
    FractionalTransformer,
    FractionalPooling,
    FractionalBatchNorm1d,
    LayerConfig,
    MLConfig
)
from hpfracc.core.definitions import FractionalOrder

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def create_sample_data(n_samples=1000, input_size=10, output_size=3):
    """Create sample training and validation data"""
    print("🔧 Creating sample data...")

    # Generate random input data
    X = torch.randn(n_samples, input_size)

    # Create target data with some non-linear relationships
    y = torch.zeros(n_samples, output_size)
    y[:, 0] = torch.sin(X[:, 0]) + 0.1 * torch.randn(n_samples)
    y[:, 1] = X[:, 1] ** 2 + 0.1 * torch.randn(n_samples)
    y[:, 2] = torch.exp(-X[:, 2]) + 0.1 * torch.randn(n_samples)

    # Split into train and validation
    train_size = int(0.8 * n_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    print(f"✅ Created {n_samples} samples")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")

    return (X_train, y_train), (X_val, y_val)

def create_fractional_model(input_size=10, hidden_sizes=[64, 32], output_size=3, fractional_order=0.5):
    """Create a fractional neural network model"""
    print(f"🧠 Creating fractional neural network with α={fractional_order}...")

    model = FractionalNeuralNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        fractional_order=FractionalOrder(fractional_order),
        activation="relu",
        dropout=0.1
    )

    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    return model

def train_model(model, train_data, val_data, epochs=20, lr=0.001):
    """Train the fractional neural network"""
    print("🚀 Starting model training...")

    X_train, y_train = train_data
    X_val, y_val = val_data

    # Setup loss function and optimizer
    criterion = FractionalMSELoss(fractional_order=FractionalOrder(0.5), method="RL")
    optimizer = FractionalAdam(
        model.parameters(),
        lr=lr,
        fractional_order=FractionalOrder(0.5),
        method="RL",
        use_fractional=True
    )

    # Training loop
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train, use_fractional=True, method="RL")
        loss = criterion(outputs, y_train, use_fractional=True)
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val, use_fractional=True, method="RL")
            val_loss = criterion(val_outputs, y_val, use_fractional=True)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss: {loss.item():.6f}, "
                  f"Val Loss: {val_loss.item():.6f}")

    print("✅ Training completed!")
    return train_losses, val_losses

def evaluate_model(model, val_data):
    """Evaluate the trained model"""
    print("📊 Evaluating model performance...")

    X_val, y_val = val_data
    model.eval()

    with torch.no_grad():
        predictions = model(X_val, use_fractional=True, method="RL")

        # Calculate metrics
        mse = nn.functional.mse_loss(predictions, y_val)
        mae = nn.functional.l1_loss(predictions, y_val)

        # Calculate R² score
        ss_res = torch.sum((y_val - predictions) ** 2)
        ss_tot = torch.sum((y_val - y_val.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)

    print(f"✅ Evaluation Results:")
    print(f"   Mean Squared Error: {mse.item():.6f}")
    print(f"   Mean Absolute Error: {mae.item():.6f}")
    print(f"   R² Score: {r2.item():.6f}")

    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'r2': r2.item()
    }

def demonstrate_fractional_attention():
    """Demonstrate fractional attention mechanism"""
    print("\n🧠 Demonstrating Fractional Attention...")

    # Create sample data
    batch_size, seq_len, d_model = 2, 10, 16
    x = torch.randn(batch_size, seq_len, d_model)

    # Create fractional attention layer
    attention = FractionalAttention(
        d_model=d_model,
        n_heads=4,
        fractional_order=FractionalOrder(0.5),
        dropout=0.1
    )

    # Apply fractional attention
    output = attention(x, method="RL")

    print(f"✅ Fractional Attention Applied:")
    print(f"   Input Shape: {x.shape}")
    print(f"   Output Shape: {output.shape}")
    print(f"   Fractional Order: {attention.fractional_order.alpha}")

    return attention

def demonstrate_fractional_layers():
    """Demonstrate fractional neural network layers"""
    print("\n🔧 Demonstrating Fractional Layers...")

    # Create configuration for fractional layers
    config = LayerConfig(
        fractional_order=FractionalOrder(0.5),
        backend="torch" 
    )

    # Demonstrate 1D Convolution
    conv1d = FractionalConv1D(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        config=config
    )

    # Demonstrate 2D Convolution
    conv2d = FractionalConv2D(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        config=config
    )

    # Demonstrate LSTM
    lstm = FractionalLSTM(
        input_size=10,
        hidden_size=32,
        config=config
    )

    # Demonstrate Transformer
    transformer = FractionalTransformer(
        d_model=64,
        nhead=8,
        config=config
    )

    # Demonstrate Pooling
    pooling = FractionalPooling(
        kernel_size=2,
        config=config
    )

    # Demonstrate Batch Normalization
    batchnorm = FractionalBatchNorm1d(
        num_features=64,
        config=config
    )

    print("✅ Fractional Layers Created Successfully!")

    # Test forward pass with sample data
    try:
        # Test Conv1D
        x1d = torch.randn(1, 3, 10)
        out1d = conv1d(x1d)
        print(f"   Conv1D Input: {x1d.shape} -> Output: {out1d.shape}")

        # Test Conv2D
        x2d = torch.randn(1, 3, 8, 8)
        out2d = conv2d(x2d)
        print(f"   Conv2D Input: {x2d.shape} -> Output: {out2d.shape}")

        # Test LSTM
        x_lstm = torch.randn(5, 1, 10)  # (seq_len, batch, input_size)
        out_lstm, (h, c) = lstm(x_lstm)
        print(f"   LSTM Input: {x_lstm.shape} -> Output: {out_lstm.shape}")

        # Test BatchNorm
        x_bn = torch.randn(1, 64, 10)
        out_bn = batchnorm(x_bn)
        print(f"   BatchNorm Input: {x_bn.shape} -> Output: {out_bn.shape}")

        print("✅ All fractional layers working correctly!")

    except Exception as e:
        print(f"❌ Error testing layers: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main demonstration function"""
    print("🚀 Starting hpfracc ML Integration Demo")
    print("=" * 50)

    # Setup logging
    setup_logging()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # 1. Create sample data
        train_data, val_data = create_sample_data()

        # 2. Create and train fractional model
        model = create_fractional_model()
        train_losses, val_losses = train_model(model, train_data, val_data)

        # 3. Evaluate model
        evaluate_model(model, val_data)

        # 4. Demonstrate fractional attention
        demonstrate_fractional_attention()

        # 5. Demonstrate fractional layers
        demonstrate_fractional_layers()

        print("\n" + "=" * 50)
        print("🎉 ML Integration Demo Completed Successfully!")
        print("\n📋 Summary of Features Demonstrated:")
        print("   ✅ Fractional Neural Networks")
        print("   ✅ Fractional Attention Mechanisms")
        print("   ✅ Fractional Convolutional Layers")
        print("   ✅ Fractional LSTM Layers")
        print("   ✅ Fractional Loss Functions")
        print("   ✅ Fractional Optimizers")

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        logging.error(f"Demo failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
