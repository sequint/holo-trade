import pytest
import numpy as np
import torch
import os
from research.train import load_and_clean_data, build_lstm_tensors, MarketLSTM

# --- FIXTURES (Reusable test setups) ---
@pytest.fixture
def mock_csv_data(tmp_path):
    """Creates a temporary mock CSV file to test pipeline ingestion."""
    csv_file = tmp_path / "mock_market.csv"
    
    # Header line with 28 columns total: id, 26 features, 1 target label
    header = "id," + ",".join([f"f{i}" for i in range(26)]) + ",target\n"
    
    # Row 1: Normal data
    row1 = "1," + ",".join([str(float(i)) for i in range(26)]) + ",1\n"
    # Row 2: Contains missing values (blank strings) to test cleaning logic
    row2 = "2,1.5," + ",".join(["" for _ in range(24)]) + ",2.5,0\n"
    # Rows 3-25: Filler rows to ensure we have enough rows to satisfy SEQUENCE_LENGTH=20
    filler_rows = ""
    for idx in range(3, 30):
        filler_rows += f"{idx}," + ",".join(["1.0" for _ in range(26)]) + ",1\n"
        
    csv_file.write_text(header + row1 + row2 + filler_rows)
    return str(csv_file)


# --- MILESTONE 1 TESTS: DATA PIPELINE ---
def test_load_and_clean_data(mock_csv_data):
    """Verifies that disk reading works and missing fields are replaced with 0.0."""
    features, labels = load_and_clean_data(mock_csv_data)
    
    # Check that ID column was stripped (should be 26 columns, not 27 or 28)
    assert features.shape[1] == 26
    # Verify that the NaN values in row 2 were cleanly swapped to 0.0
    assert not np.isnan(features).any()
    assert features[1, 5] == 0.0 


def test_build_lstm_tensors():
    """Verifies that flat matrices are properly transformed into 3D sequential sliding windows."""
    # Generate mock clean arrays: 50 rows, 26 features
    mock_features = np.ones((50, 26))
    mock_labels = np.ones((50,))
    seq_len = 20
    
    X_tensor, y_tensor = build_lstm_tensors(mock_features, mock_labels, seq_len)
    
    # Expected windows: total_rows - seq_len = 50 - 20 = 30
    assert X_tensor.shape == torch.Size([30, seq_len, 26])
    assert y_tensor.shape == torch.Size([30])
    assert X_tensor.dtype == torch.float32
    assert y_tensor.dtype == torch.int64


# --- MILESTONE 2 TESTS: MODEL ARCHITECTURE ---
def test_market_lstm_forward_pass():
    """Verifies that the network completes a forward pass without throwing dimension errors."""
    batch_size = 8
    seq_len = 20
    input_dim = 26
    hidden_dim = 32
    output_dim = 3
    
    model = MarketLSTM(input_dim, hidden_dim, num_layers=2, output_dim=output_dim)
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    # Run forward pass
    with torch.no_grad():
        output = model(dummy_input)
        
    # The output matrix MUST match [batch_size, output_dim]
    assert output.shape == torch.Size([batch_size, output_dim])