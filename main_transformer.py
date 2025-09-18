import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from data_loader import download_data
from transformer_model import TransformerModel

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def main():
    # --- 1. Parameters ---
    TICKER = 'AAPL'
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'
    LOOK_BACK = 60
    TRAIN_SPLIT = 0.8
    
    # Model Hyperparameters
    EPOCHS = 20
    LEARNING_RATE = 0.001
    D_MODEL = 64
    NHEAD = 4
    NUM_ENCODER_LAYERS = 2
    DIM_FEEDFORWARD = 256
    DROPOUT = 0.1

    # --- 2. Load and Preprocess Data ---
    print(f"Loading data for {TICKER}...")
    stock_data = download_data(TICKER, START_DATE, END_DATE)
    if stock_data is None:
        return

    close_prices = stock_data['Close'].values.reshape(-1, 1)

    # Corrected Scaling: Split data BEFORE scaling
    train_size = int(len(close_prices) * TRAIN_SPLIT)
    train_data_raw = close_prices[:train_size]
    test_data_raw = close_prices[train_size:]

    # Fit scaler ONLY on training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(train_data_raw)

    # Transform both sets
    train_data_scaled = scaler.transform(train_data_raw)
    test_data_scaled = scaler.transform(test_data_raw)

    # --- 3. Create Sequences and Tensors ---
    X_train, y_train = create_sequences(train_data_scaled, LOOK_BACK)
    X_test, y_test = create_sequences(test_data_scaled, LOOK_BACK)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # --- 4. Initialize Model, Loss, and Optimizer ---
    model = TransformerModel(
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    )
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 5. Train the Model ---
    print("\nStarting Transformer model training...")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_function(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}')
    print("Training finished.")

    # --- 6. Evaluate the Model ---
    print("\nEvaluating model...")
    model.eval()
    with torch.no_grad():
        test_predictions_scaled = model(X_test)

    # Inverse transform the predictions to get actual price values
    test_predictions = scaler.inverse_transform(test_predictions_scaled.numpy())
    
    # The actual test values are the latter part of the original unscaled data
    y_test_actual = test_data_raw[LOOK_BACK:]


    # --- 7. Plot the Results ---
    print("Plotting results...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(14, 7))
    
    # The index for the test predictions
    test_data_index = stock_data.index[train_size + LOOK_BACK:]

    # Make sure the lengths match
    if len(test_data_index) != len(test_predictions):
        print("Warning: Length of test data index and predictions do not match.")
        # Fallback to ensure plotting works, though it might be misaligned
        test_data_index = stock_data.index[-len(test_predictions):]


    plt.plot(stock_data.index, close_prices, label='Historical Prices', color='royalblue', alpha=0.5)
    plt.plot(test_data_index, y_test_actual, label='Actual Test Prices', color='green')
    plt.plot(test_data_index, test_predictions, label='Predicted Test Prices (Transformer)', color='purple', linestyle='--')
    
    plt.title(f'{TICKER} Stock Price Prediction (Transformer Model)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    plot_filename = 'prediction_plot_transformer.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.close(fig)

if __name__ == "__main__":
    main()