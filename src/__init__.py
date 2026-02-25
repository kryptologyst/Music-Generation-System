"""Music Generation System - A modern, research-focused music generation system."""

__version__ = "1.0.0"
__author__ = "Music Generation Team"
__email__ = "contact@musicgeneration.ai"

from .models.music_generator import MusicTransformer, MusicLSTM, create_model
from .data.midi_dataset import MIDIDataset, MIDITokenizer, create_data_loaders
from .train.trainer import MusicTrainer
from .eval.evaluator import MusicEvaluator
from .metrics.metrics import MusicMetrics
from .utils.utils import set_seed, get_device, save_checkpoint, load_checkpoint

__all__ = [
    "MusicTransformer",
    "MusicLSTM", 
    "create_model",
    "MIDIDataset",
    "MIDITokenizer",
    "create_data_loaders",
    "MusicTrainer",
    "MusicEvaluator",
    "MusicMetrics",
    "set_seed",
    "get_device",
    "save_checkpoint",
    "load_checkpoint"
]
