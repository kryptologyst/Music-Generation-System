"""Evaluation metrics for music generation."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any
import logging
from collections import Counter
import mir_eval
from mir_eval.melody import evaluate as melody_evaluate
from mir_eval.chord import evaluate as chord_evaluate

logger = logging.getLogger(__name__)


class MusicMetrics:
    """Class for calculating music generation metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass
        
    def calculate_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive music generation metrics.
        
        Args:
            predictions: Predicted token sequences
            targets: Target token sequences
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic accuracy metrics
        metrics.update(self._calculate_accuracy_metrics(predictions, targets))
        
        # Music-specific metrics
        metrics.update(self._calculate_music_metrics(predictions, targets))
        
        # Diversity metrics
        metrics.update(self._calculate_diversity_metrics(predictions))
        
        return metrics
        
    def _calculate_accuracy_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate basic accuracy metrics.
        
        Args:
            predictions: Predicted sequences
            targets: Target sequences
            
        Returns:
            Dictionary of accuracy metrics
        """
        # Flatten arrays for token-level accuracy
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Remove padding tokens (assuming 0 is padding)
        mask = target_flat != 0
        pred_flat = pred_flat[mask]
        target_flat = target_flat[mask]
        
        # Token accuracy
        token_accuracy = np.mean(pred_flat == target_flat)
        
        # Perplexity (approximation)
        # This is a simplified calculation - in practice, you'd use the actual logits
        unique_tokens = len(np.unique(target_flat))
        perplexity = np.exp(-np.mean(np.log(np.maximum(pred_flat == target_flat, 1e-10))))
        
        return {
            'token_accuracy': token_accuracy,
            'perplexity': perplexity
        }
        
    def _calculate_music_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate music-specific metrics.
        
        Args:
            predictions: Predicted sequences
            targets: Target sequences
            
        Returns:
            Dictionary of music metrics
        """
        metrics = {}
        
        # Pitch accuracy
        pitch_acc = self._calculate_pitch_accuracy(predictions, targets)
        metrics['pitch_accuracy'] = pitch_acc
        
        # Rhythm accuracy
        rhythm_acc = self._calculate_rhythm_accuracy(predictions, targets)
        metrics['rhythm_accuracy'] = rhythm_acc
        
        # Harmonic coherence
        harmonic_coherence = self._calculate_harmonic_coherence(predictions)
        metrics['harmonic_coherence'] = harmonic_coherence
        
        # Melodic continuity
        melodic_continuity = self._calculate_melodic_continuity(predictions)
        metrics['melodic_continuity'] = melodic_continuity
        
        return metrics
        
    def _calculate_pitch_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Calculate pitch accuracy.
        
        Args:
            predictions: Predicted sequences
            targets: Target sequences
            
        Returns:
            Pitch accuracy score
        """
        # Extract note tokens (assuming notes are in range 4-127)
        note_mask = (targets >= 4) & (targets <= 127)
        
        if np.sum(note_mask) == 0:
            return 0.0
            
        pred_notes = predictions[note_mask]
        target_notes = targets[note_mask]
        
        # Calculate accuracy for note predictions
        pitch_accuracy = np.mean(pred_notes == target_notes)
        
        return pitch_accuracy
        
    def _calculate_rhythm_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Calculate rhythm accuracy.
        
        Args:
            predictions: Predicted sequences
            targets: Target sequences
            
        Returns:
            Rhythm accuracy score
        """
        # Extract duration tokens (assuming duration tokens start with 'DUR_')
        # This is a simplified calculation - in practice, you'd need to parse the tokenizer
        duration_mask = (targets >= 128) & (targets <= 133)  # Assuming duration tokens are in this range
        
        if np.sum(duration_mask) == 0:
            return 0.0
            
        pred_durations = predictions[duration_mask]
        target_durations = targets[duration_mask]
        
        # Calculate accuracy for duration predictions
        rhythm_accuracy = np.mean(pred_durations == target_durations)
        
        return rhythm_accuracy
        
    def _calculate_harmonic_coherence(
        self,
        predictions: np.ndarray
    ) -> float:
        """Calculate harmonic coherence score.
        
        Args:
            predictions: Predicted sequences
            
        Returns:
            Harmonic coherence score
        """
        # Extract note sequences
        note_sequences = []
        for seq in predictions:
            notes = seq[(seq >= 4) & (seq <= 127)]  # Extract note tokens
            if len(notes) > 0:
                note_sequences.append(notes)
                
        if not note_sequences:
            return 0.0
            
        # Calculate harmonic intervals
        harmonic_scores = []
        for notes in note_sequences:
            if len(notes) < 2:
                continue
                
            # Calculate intervals between consecutive notes
            intervals = np.diff(notes)
            
            # Count consonant intervals (perfect fourth, fifth, octave, etc.)
            consonant_intervals = [3, 4, 7, 8, 12]  # Minor third, major third, perfect fifth, minor sixth, octave
            consonant_count = sum(1 for interval in intervals if interval in consonant_intervals)
            
            if len(intervals) > 0:
                harmonic_score = consonant_count / len(intervals)
                harmonic_scores.append(harmonic_score)
                
        return np.mean(harmonic_scores) if harmonic_scores else 0.0
        
    def _calculate_melodic_continuity(
        self,
        predictions: np.ndarray
    ) -> float:
        """Calculate melodic continuity score.
        
        Args:
            predictions: Predicted sequences
            
        Returns:
            Melodic continuity score
        """
        # Extract note sequences
        note_sequences = []
        for seq in predictions:
            notes = seq[(seq >= 4) & (seq <= 127)]  # Extract note tokens
            if len(notes) > 0:
                note_sequences.append(notes)
                
        if not note_sequences:
            return 0.0
            
        continuity_scores = []
        for notes in note_sequences:
            if len(notes) < 2:
                continue
                
            # Calculate melodic intervals
            intervals = np.diff(notes)
            
            # Count small intervals (step-wise motion)
            small_intervals = [1, 2]  # Semitone, whole tone
            step_count = sum(1 for interval in intervals if abs(interval) in small_intervals)
            
            # Count large intervals (leaps)
            large_intervals = [6, 7, 8, 9, 10, 11]  # Larger intervals
            leap_count = sum(1 for interval in intervals if abs(interval) in large_intervals)
            
            if len(intervals) > 0:
                # Prefer step-wise motion for melodic continuity
                continuity_score = step_count / (step_count + leap_count) if (step_count + leap_count) > 0 else 0.0
                continuity_scores.append(continuity_score)
                
        return np.mean(continuity_scores) if continuity_scores else 0.0
        
    def _calculate_diversity_metrics(
        self,
        predictions: np.ndarray
    ) -> Dict[str, float]:
        """Calculate diversity metrics.
        
        Args:
            predictions: Predicted sequences
            
        Returns:
            Dictionary of diversity metrics
        """
        metrics = {}
        
        # Token diversity
        all_tokens = predictions.flatten()
        unique_tokens = len(np.unique(all_tokens))
        total_tokens = len(all_tokens)
        
        metrics['token_diversity'] = unique_tokens / total_tokens if total_tokens > 0 else 0.0
        
        # Sequence diversity
        sequence_diversity = self._calculate_sequence_diversity(predictions)
        metrics['sequence_diversity'] = sequence_diversity
        
        # Note diversity
        note_diversity = self._calculate_note_diversity(predictions)
        metrics['note_diversity'] = note_diversity
        
        return metrics
        
    def _calculate_sequence_diversity(
        self,
        predictions: np.ndarray
    ) -> float:
        """Calculate sequence diversity.
        
        Args:
            predictions: Predicted sequences
            
        Returns:
            Sequence diversity score
        """
        # Convert sequences to strings for comparison
        sequence_strings = [str(seq.tolist()) for seq in predictions]
        
        # Count unique sequences
        unique_sequences = len(set(sequence_strings))
        total_sequences = len(sequence_strings)
        
        return unique_sequences / total_sequences if total_sequences > 0 else 0.0
        
    def _calculate_note_diversity(
        self,
        predictions: np.ndarray
    ) -> float:
        """Calculate note diversity.
        
        Args:
            predictions: Predicted sequences
            
        Returns:
            Note diversity score
        """
        # Extract all note tokens
        all_notes = []
        for seq in predictions:
            notes = seq[(seq >= 4) & (seq <= 127)]  # Extract note tokens
            all_notes.extend(notes)
            
        if not all_notes:
            return 0.0
            
        # Calculate note distribution
        note_counts = Counter(all_notes)
        total_notes = len(all_notes)
        
        # Calculate entropy (diversity measure)
        entropy = 0.0
        for count in note_counts.values():
            probability = count / total_notes
            entropy -= probability * np.log2(probability)
            
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(note_counts)) if len(note_counts) > 1 else 0.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
        
    def calculate_style_consistency(
        self,
        predictions: np.ndarray,
        reference_style: np.ndarray
    ) -> float:
        """Calculate style consistency with reference.
        
        Args:
            predictions: Predicted sequences
            reference_style: Reference style sequences
            
        Returns:
            Style consistency score
        """
        # Extract note sequences from predictions
        pred_notes = []
        for seq in predictions:
            notes = seq[(seq >= 4) & (seq <= 127)]
            if len(notes) > 0:
                pred_notes.extend(notes)
                
        # Extract note sequences from reference
        ref_notes = []
        for seq in reference_style:
            notes = seq[(seq >= 4) & (seq <= 127)]
            if len(notes) > 0:
                ref_notes.extend(notes)
                
        if not pred_notes or not ref_notes:
            return 0.0
            
        # Calculate note distribution similarity
        pred_counts = Counter(pred_notes)
        ref_counts = Counter(ref_notes)
        
        # Get all unique notes
        all_notes = set(pred_counts.keys()) | set(ref_counts.keys())
        
        # Calculate KL divergence
        kl_div = 0.0
        for note in all_notes:
            pred_prob = pred_counts.get(note, 0) / len(pred_notes)
            ref_prob = ref_counts.get(note, 0) / len(ref_notes)
            
            if ref_prob > 0 and pred_prob > 0:
                kl_div += pred_prob * np.log2(pred_prob / ref_prob)
                
        # Convert to similarity score (lower KL divergence = higher similarity)
        similarity = np.exp(-kl_div)
        
        return similarity
