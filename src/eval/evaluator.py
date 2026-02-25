"""Evaluation module for music generation models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np
from tqdm import tqdm

from ..metrics.metrics import MusicMetrics

logger = logging.getLogger(__name__)


class MusicEvaluator:
    """Evaluator class for music generation models."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        tokenizer: Any,
        device: torch.device
    ):
        """Initialize evaluator.
        
        Args:
            model: Trained PyTorch model
            test_loader: Test data loader
            tokenizer: MIDI tokenizer
            device: Device to evaluate on
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.device = device
        self.metrics = MusicMetrics()
        
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model.
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting evaluation...")
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for input_ids, target_ids in tqdm(self.test_loader, desc="Evaluating"):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward'):
                    logits = self.model(input_ids)
                else:
                    logits, _ = self.model(input_ids)
                    
                # Calculate loss
                loss = nn.CrossEntropyLoss(ignore_index=0)(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Store for metrics calculation
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target_ids.cpu().numpy())
                
        # Calculate metrics
        avg_loss = total_loss / num_batches
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate additional metrics
        metrics = self.metrics.calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss
        
        logger.info("Evaluation completed")
        logger.info(f"Test Loss: {avg_loss:.4f}")
        for metric_name, value in metrics.items():
            if metric_name != 'loss':
                logger.info(f"{metric_name}: {value:.4f}")
                
        return metrics
        
    def generate_samples(
        self,
        num_samples: int = 5,
        max_length: int = 500,
        temperature: float = 0.8
    ) -> List[List[int]]:
        """Generate sample sequences.
        
        Args:
            num_samples: Number of samples to generate
            max_length: Maximum length of generated sequences
            temperature: Sampling temperature
            
        Returns:
            List of generated sequences
        """
        logger.info(f"Generating {num_samples} samples...")
        
        self.model.eval()
        samples = []
        
        # Get a random seed from the test set
        seed_batch = next(iter(self.test_loader))
        seed_sequences = seed_batch[0][:num_samples].to(self.device)
        
        with torch.no_grad():
            for i in range(num_samples):
                seed = seed_sequences[i:i+1]
                
                if hasattr(self.model, 'generate'):
                    generated = self.model.generate(
                        seed,
                        max_length=max_length,
                        temperature=temperature
                    )
                else:
                    # Fallback generation method
                    generated = self._simple_generate(
                        seed,
                        max_length=max_length,
                        temperature=temperature
                    )
                    
                samples.append(generated[0].cpu().numpy().tolist())
                
        return samples
        
    def _simple_generate(
        self,
        seed: torch.Tensor,
        max_length: int = 500,
        temperature: float = 0.8
    ) -> torch.Tensor:
        """Simple generation method for models without generate method.
        
        Args:
            seed: Initial sequence
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated sequence
        """
        generated = seed.clone()
        
        for _ in range(max_length):
            # Get logits for the last position
            if hasattr(self.model, 'forward'):
                logits = self.model(generated)
            else:
                logits, _ = self.model(generated)
                
            logits = logits[:, -1, :] / temperature
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for end token
            if (next_token == 2).all():
                break
                
        return generated
