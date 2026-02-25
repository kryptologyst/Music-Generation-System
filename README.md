# Music Generation System

Research-focused music generation system using deep learning techniques. This project implements state-of-the-art models for generating musical sequences from MIDI data.

## Features

- **Modern Architecture**: Transformer-based and LSTM models for music generation
- **MIDI Processing**: Comprehensive MIDI file handling and preprocessing
- **Evaluation Metrics**: Multiple metrics for assessing generated music quality
- **Interactive Demo**: Streamlit-based web interface for music generation
- **Research Focus**: Designed for academic research and educational purposes

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Music-Generation-System.git
cd Music-Generation-System

# Install dependencies
pip install -r requirements.txt

# Run the demo
streamlit run demo/app.py
```

### Basic Usage

```python
from src.models.music_generator import MusicGenerator
from src.data.midi_dataset import MIDIDataset

# Load dataset
dataset = MIDIDataset("data/raw/midi_files/")

# Initialize model
generator = MusicGenerator(vocab_size=dataset.vocab_size)

# Train model
generator.train(dataset, epochs=100)

# Generate music
generated_sequence = generator.generate(seed_length=100, max_length=500)
```

## Dataset Schema

The system expects MIDI files in the following structure:

```
data/
├── raw/
│   ├── midi_files/          # Directory containing MIDI files
│   └── meta.csv             # Metadata file with columns: id, path, genre, tempo, etc.
└── processed/               # Processed data (auto-generated)
```

### Metadata Format

The `meta.csv` file should contain:
- `id`: Unique identifier
- `path`: Path to MIDI file
- `genre`: Music genre (optional)
- `tempo`: Tempo in BPM (optional)
- `duration`: Duration in seconds (optional)

## Training and Evaluation

### Training

```bash
python scripts/train.py --config configs/train_config.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
```

## Demo

Launch the interactive demo:

```bash
streamlit run demo/app.py
```

The demo allows you to:
- Upload MIDI files for training
- Generate new music sequences
- Adjust generation parameters
- Listen to generated music
- Export results as MIDI files

## Metrics

The system evaluates generated music using:

- **Pitch Accuracy**: Correctness of generated pitches
- **Rhythm Accuracy**: Timing and rhythm pattern accuracy
- **Harmonic Coherence**: Chord progression quality
- **Melodic Continuity**: Smoothness of melodic lines
- **Style Consistency**: Adherence to training style

## Model Architecture

### LSTM Model
- Multi-layer LSTM with dropout
- Sequence-to-sequence generation
- Attention mechanism for long sequences

### Transformer Model
- Multi-head self-attention
- Positional encoding for temporal relationships
- Causal masking for autoregressive generation

## Configuration

Configuration files are located in `configs/`:

- `train_config.yaml`: Training parameters
- `model_config.yaml`: Model architecture settings
- `data_config.yaml`: Data processing options

## Privacy and Ethics

**IMPORTANT**: This system is designed for research and educational purposes only. Please read the [DISCLAIMER.md](DISCLAIMER.md) for important information about ethical use, privacy considerations, and limitations.

## Limitations

- Generated music quality depends on training data quality and quantity
- May not capture complex musical structures or emotions
- Not suitable for commercial use without proper licensing
- Generated content should be used responsibly and ethically

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{music_generation_system,
  title={Music Generation System},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Music-Generation-System}
}
```

## Contact

For questions or support, please open an issue on GitHub or contact the maintainers.
# Music-Generation-System
