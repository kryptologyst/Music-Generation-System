"""Data processing utilities for MIDI files."""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import mido
import pretty_midi
from music21 import converter, stream, note, chord, duration, tempo, key, meter
import random

logger = logging.getLogger(__name__)


class MIDITokenizer:
    """Tokenizer for MIDI sequences."""
    
    def __init__(self, vocab_size: int = 128):
        """Initialize tokenizer.
        
        Args:
            vocab_size: Size of vocabulary
        """
        self.vocab_size = vocab_size
        self.note_to_int = {}
        self.int_to_note = {}
        self.special_tokens = {
            'PAD': 0,
            'START': 1,
            'END': 2,
            'REST': 3
        }
        self._build_vocab()
        
    def _build_vocab(self) -> None:
        """Build vocabulary mapping."""
        # Add special tokens
        for token, idx in self.special_tokens.items():
            self.note_to_int[token] = idx
            self.int_to_note[idx] = token
            
        # Add note tokens (MIDI note numbers 21-108)
        for midi_num in range(21, 109):  # A0 to C8
            note_name = pretty_midi.note_number_to_name(midi_num)
            idx = len(self.note_to_int)
            self.note_to_int[note_name] = idx
            self.int_to_note[idx] = note_name
            
        # Add duration tokens
        durations = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]  # eighth, quarter, half, whole, etc.
        for dur in durations:
            idx = len(self.note_to_int)
            self.note_to_int[f'DUR_{dur}'] = idx
            self.int_to_note[idx] = f'DUR_{dur}'
            
        # Add velocity tokens
        velocities = [32, 64, 96, 127]  # pp, mp, mf, ff
        for vel in velocities:
            idx = len(self.note_to_int)
            self.note_to_int[f'VEL_{vel}'] = idx
            self.int_to_note[idx] = f'VEL_{vel}'
            
        logger.info(f"Built vocabulary with {len(self.note_to_int)} tokens")
        
    def encode(self, midi_sequence: List[Dict[str, Any]]) -> List[int]:
        """Encode MIDI sequence to token IDs.
        
        Args:
            midi_sequence: List of MIDI events
            
        Returns:
            List of token IDs
        """
        tokens = [self.special_tokens['START']]
        
        for event in midi_sequence:
            if event['type'] == 'note_on':
                note_name = pretty_midi.note_number_to_name(event['note'])
                if note_name in self.note_to_int:
                    tokens.append(self.note_to_int[note_name])
                    
                # Add duration token
                dur_key = f'DUR_{event["duration"]}'
                if dur_key in self.note_to_int:
                    tokens.append(self.note_to_int[dur_key])
                    
                # Add velocity token
                vel_key = f'VEL_{event["velocity"]}'
                if vel_key in self.note_to_int:
                    tokens.append(self.note_to_int[vel_key])
                    
            elif event['type'] == 'rest':
                tokens.append(self.special_tokens['REST'])
                
        tokens.append(self.special_tokens['END'])
        return tokens
        
    def decode(self, token_ids: List[int]) -> List[Dict[str, Any]]:
        """Decode token IDs to MIDI sequence.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            List of MIDI events
        """
        events = []
        i = 0
        
        while i < len(token_ids):
            token_id = token_ids[i]
            
            if token_id == self.special_tokens['START'] or token_id == self.special_tokens['END']:
                i += 1
                continue
            elif token_id == self.special_tokens['REST']:
                events.append({'type': 'rest', 'duration': 0.25})
                i += 1
            else:
                token = self.int_to_note[token_id]
                
                if token.startswith('DUR_'):
                    # Duration token - skip for now
                    i += 1
                elif token.startswith('VEL_'):
                    # Velocity token - skip for now
                    i += 1
                else:
                    # Note token
                    try:
                        note_num = pretty_midi.note_name_to_number(token)
                        events.append({
                            'type': 'note_on',
                            'note': note_num,
                            'velocity': 64,
                            'duration': 0.25
                        })
                    except ValueError:
                        logger.warning(f"Invalid note token: {token}")
                    i += 1
                    
        return events


class MIDIDataset(Dataset):
    """Dataset for MIDI files."""
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 100,
        overlap: int = 50,
        tokenizer: Optional[MIDITokenizer] = None
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing MIDI files
            sequence_length: Length of input sequences
            overlap: Overlap between sequences
            tokenizer: MIDI tokenizer
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.tokenizer = tokenizer or MIDITokenizer()
        
        self.midi_files = self._find_midi_files()
        self.sequences = self._load_sequences()
        
        logger.info(f"Loaded {len(self.sequences)} sequences from {len(self.midi_files)} MIDI files")
        
    def _find_midi_files(self) -> List[Path]:
        """Find all MIDI files in the directory."""
        midi_extensions = ['.mid', '.midi', '.MID', '.MIDI']
        midi_files = []
        
        for ext in midi_extensions:
            midi_files.extend(self.data_dir.glob(f'**/*{ext}'))
            
        return midi_files
        
    def _load_sequences(self) -> List[List[int]]:
        """Load and tokenize all MIDI sequences."""
        sequences = []
        
        for midi_file in self.midi_files:
            try:
                midi_events = self._parse_midi_file(midi_file)
                if midi_events:
                    tokens = self.tokenizer.encode(midi_events)
                    if len(tokens) > self.sequence_length:
                        sequences.append(tokens)
            except Exception as e:
                logger.warning(f"Failed to load {midi_file}: {e}")
                
        return sequences
        
    def _parse_midi_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse MIDI file to extract note events."""
        try:
            midi_data = pretty_midi.PrettyMIDI(str(file_path))
            events = []
            
            for instrument in midi_data.instruments:
                if instrument.is_drum:
                    continue
                    
                for note in instrument.notes:
                    events.append({
                        'type': 'note_on',
                        'note': note.pitch,
                        'velocity': note.velocity,
                        'start': note.start,
                        'end': note.end,
                        'duration': note.end - note.start
                    })
                    
            # Sort by start time
            events.sort(key=lambda x: x['start'])
            
            return events
            
        except Exception as e:
            logger.error(f"Error parsing MIDI file {file_path}: {e}")
            return []
            
    def __len__(self) -> int:
        """Return dataset length."""
        total_sequences = 0
        for seq in self.sequences:
            if len(seq) > self.sequence_length:
                total_sequences += (len(seq) - self.sequence_length) // self.overlap + 1
        return total_sequences
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (input_sequence, target_sequence)
        """
        # Find which sequence and position this index corresponds to
        current_idx = 0
        for seq in self.sequences:
            if len(seq) <= self.sequence_length:
                continue
                
            num_subsequences = (len(seq) - self.sequence_length) // self.overlap + 1
            if current_idx + num_subsequences > idx:
                # This sequence contains our target
                local_idx = idx - current_idx
                start_pos = local_idx * self.overlap
                end_pos = start_pos + self.sequence_length + 1
                
                if end_pos <= len(seq):
                    subseq = seq[start_pos:end_pos]
                    input_seq = torch.tensor(subseq[:-1], dtype=torch.long)
                    target_seq = torch.tensor(subseq[1:], dtype=torch.long)
                    return input_seq, target_seq
                    
            current_idx += num_subsequences
            
        # Fallback - return random sequence
        seq = random.choice(self.sequences)
        if len(seq) > self.sequence_length:
            start = random.randint(0, len(seq) - self.sequence_length - 1)
            subseq = seq[start:start + self.sequence_length + 1]
            input_seq = torch.tensor(subseq[:-1], dtype=torch.long)
            target_seq = torch.tensor(subseq[1:], dtype=torch.long)
            return input_seq, target_seq
        else:
            # Pad sequence if too short
            padded_seq = seq + [self.tokenizer.special_tokens['PAD']] * (self.sequence_length + 1 - len(seq))
            input_seq = torch.tensor(padded_seq[:-1], dtype=torch.long)
            target_seq = torch.tensor(padded_seq[1:], dtype=torch.long)
            return input_seq, target_seq


def create_data_loaders(
    dataset: MIDIDataset,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders.
    
    Args:
        dataset: MIDI dataset
        batch_size: Batch size
        train_split: Training set proportion
        val_split: Validation set proportion
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
