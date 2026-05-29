import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time

# Config Constants
INPUT_FEATURES = 27
SEQUENCE_LENGTH = 20
BATCH_SIZE = 32

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

# Main execution
if __name__ == "__main__":
    DATA_PATH = "../data/train.csv"

    print("Loading and preprossesing market data...")

    X_features, y_labels = load_and_clean_data(DATA_PATH)
    X, y = build_lstm_tensors(X_features, y_labels, SEQUENCE_LENGTH)

    print("\n--- DATA PIPELINE VERIFICATION ---")
    print(f"X Tensor Shape (Expecting 3D): {X.shape}")
    print(f"y Tensor Shape (Expecting 1D): {y.shape}")

    # Combine features and targets into a dataset
    dataset = TensorDataset(X, y)

    # Build dataloader factory belt
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Get mini-set from factory belt to verify small dataset shape
    first_batch_X, first_batch_y = next(iter(train_loader))

    print("\n--- DATALOADER MINI-BATCH VERIFICATION ---")
    print(f"Mini-batch X shape (Expecting [32, 20, 26]): {first_batch_X.shape}")
    print(f"Mini-batch y shape (Expecting [32]): {first_batch_y.shape}")