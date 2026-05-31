import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time

# Config Constants
INPUT_FEATURES = 27
SEQUENCE_LENGTH = 20
BATCH_SIZE = 32

# --- DATA INGESTION --- #
def load_and_clean_data(data_path):
    # Load the data (skip header in csv file)
    print("  └─ Reading raw CSV from disk... ", end="", flush=True)
    start_time = time.time()
    raw_data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    print(f"Done! ({time.time() - start_time:.2f}s)")

    # Seperate features from target class and drop id col and handle null values
    print("  └─ Cleaning missing fields (NaN -> 0.0)... ", end="", flush=True)
    features = raw_data[:, 1:-1]
    labels = raw_data[:, -1]
    features = np.nan_to_num(features, nan=0.0)
    print("Done!")

    return features, labels

def build_lstm_tensors(features, labels, sequence_length):
    windows = []
    targets = []
    total_iterations = len(features) - sequence_length

    # Create historic market data sliding window
    print("  └─ Slicing sequential time-windows:")
    for i in tqdm(range(total_iterations), desc="     Processing Windows", unit=" window"):
        # Create a window of transactions starting from the current position
        window = features[i : i + sequence_length]
        windows.append(window)

        # Store the corresponding class target at the end of current window for timeline causality
        target = labels[i + sequence_length - 1]
        targets.append(target)
    
    # Covnert arrays to tensors
    print("  └─ Converting nested structures to PyTorch Tensors... ", end="", flush=True)
    X_tensor = torch.tensor(np.array(windows), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(targets), dtype=torch.long)
    print("Done!")
    
    return X_tensor, y_tensor

# --- LSTM MODEL ARCHITECTURE ---#
class MarketLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.2):
        super(MarketLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        last_time_step_out = out[:, -1, :]
        prediction = self.fc(last_time_step_out)

        return prediction

# Main execution
if __name__ == "__main__":
    DATA_PATH = "data/train.csv"

    print("Loading and preprossesing market data...")

    X_features, y_labels = load_and_clean_data(DATA_PATH)
    X, y = build_lstm_tensors(X_features, y_labels, SEQUENCE_LENGTH)

    print("\n--- DATA PIPELINE VERIFICATION ---")
    print(f"X Tensor Shape (Expecting 3D): {X.shape}")
    print(f"y Tensor Shape (Expecting 1D): {y.shape}")

    print("\nInitializing LSTM Market Model...")
    HIDDEN_DIM = 64      
    NUM_LAYERS = 2       
    OUTPUT_DIM = 3       
    DROPOUT_PROB = 0.2

    model = MarketMarketLSTM = MarketLSTM(
        input_dim=X.shape[2],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_dim=OUTPUT_DIM,
        dropout_prob=DROPOUT_PROB
    )

    # Temp dummy input tensor to validate model
    dummy_input = torch.rand(BATCH_SIZE, SEQUENCE_LENGTH, X.shape[2])

    print("  └─ Executing test forward pass with dummy tensor... ", end="", flush=True)
    with torch.no_grad():
        dummy_output = model(dummy_input)
    print("Done!")
    
    print("\n--- MODEL ARCHITECTURE VERIFICATION ---")
    print(f"  └─ Dummy Input Shape Given:  {dummy_input.shape}")
    print(f"  └─ Model Output Shape Vector: {dummy_output.shape}")