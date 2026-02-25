#!/usr/bin/env python3
"""Simple test script for the music generation system."""

import torch
import numpy as np
from src.models.music_generator import MusicTransformer, MusicLSTM
from src.data.midi_dataset import MIDITokenizer
from src.utils.utils import set_seed, get_device

def test_model_creation():
    """Test model creation and basic functionality."""
    print("Testing model creation...")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Test Transformer model
    print("Creating Transformer model...")
    transformer = MusicTransformer(
        vocab_size=128,
        d_model=256,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    )
    transformer = transformer.to(device)
    
    # Test LSTM model
    print("Creating LSTM model...")
    lstm = MusicLSTM(
        vocab_size=128,
        embedding_dim=128,
        hidden_dim=256,
        n_layers=2,
        dropout=0.1
    )
    lstm = lstm.to(device)
    
    # Test forward pass
    print("Testing forward pass...")
    batch_size = 2
    seq_length = 50
    
    # Create dummy input
    input_ids = torch.randint(0, 128, (batch_size, seq_length)).to(device)
    
    # Test Transformer
    transformer.eval()
    with torch.no_grad():
        transformer_output = transformer(input_ids)
        print(f"Transformer output shape: {transformer_output.shape}")
    
    # Test LSTM
    lstm.eval()
    with torch.no_grad():
        lstm_output, hidden = lstm(input_ids)
        print(f"LSTM output shape: {lstm_output.shape}")
    
    # Test generation
    print("Testing generation...")
    seed = torch.randint(0, 128, (1, 10)).to(device)
    
    with torch.no_grad():
        if hasattr(transformer, 'generate'):
            generated = transformer.generate(seed, max_length=50, temperature=0.8)
            print(f"Generated sequence length: {generated.shape[1]}")
    
    print("All tests passed!")


def test_tokenizer():
    """Test MIDI tokenizer functionality."""
    print("Testing MIDI tokenizer...")
    
    tokenizer = MIDITokenizer()
    
    # Test encoding
    midi_sequence = [
        {'type': 'note_on', 'note': 60, 'velocity': 64, 'duration': 0.25},
        {'type': 'note_on', 'note': 64, 'velocity': 64, 'duration': 0.25},
        {'type': 'note_on', 'note': 67, 'velocity': 64, 'duration': 0.25},
        {'type': 'rest', 'duration': 0.25}
    ]
    
    tokens = tokenizer.encode(midi_sequence)
    print(f"Encoded tokens: {tokens[:10]}...")  # Show first 10 tokens
    
    # Test decoding
    decoded_events = tokenizer.decode(tokens)
    print(f"Decoded events: {len(decoded_events)} events")
    
    print("Tokenizer test passed!")


def test_metrics():
    """Test metrics calculation."""
    print("Testing metrics calculation...")
    
    from src.metrics.metrics import MusicMetrics
    
    # Create dummy predictions and targets
    predictions = np.random.randint(0, 128, (10, 100))
    targets = np.random.randint(0, 128, (10, 100))
    
    metrics = MusicMetrics()
    results = metrics.calculate_metrics(predictions, targets)
    
    print("Metrics calculated:")
    for metric_name, value in results.items():
        print(f"  {metric_name}: {value:.4f}")
    
    print("Metrics test passed!")


if __name__ == "__main__":
    print("Running Music Generation System Tests")
    print("=" * 50)
    
    try:
        test_model_creation()
        print()
        test_tokenizer()
        print()
        test_metrics()
        print()
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
