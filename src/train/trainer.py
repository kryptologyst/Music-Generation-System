"""Training utilities for music generation models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

from ..utils.utils import EarlyStopping, save_checkpoint, load_checkpoint, format_time
from ..metrics.metrics import MusicMetrics

logger = logging.getLogger(__name__)


class MusicTrainer:
    """Trainer class for music generation models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        checkpoint_dir: str = "checkpoints"
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        
        # Initialize metrics
        self.metrics = MusicMetrics()
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 10),
            min_delta=config.get('min_delta', 0.001)
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')
        learning_rate = optimizer_config.get('learning_rate', 0.001)
        weight_decay = optimizer_config.get('weight_decay', 0.01)
        
        if optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100)
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            return None
            
    def train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if isinstance(self.model, nn.TransformerEncoder):
                # Transformer model
                logits = self.model(input_ids)
            else:
                # LSTM model
                logits, _ = self.model(input_ids)
                
            # Calculate loss
            loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip_norm']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        return total_loss / num_batches
        
    def validate_epoch(self) -> float:
        """Validate for one epoch.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, target_ids in tqdm(self.val_loader, desc="Validation"):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # Forward pass
                if isinstance(self.model, nn.TransformerEncoder):
                    logits = self.model(input_ids)
                else:
                    logits, _ = self.model(input_ids)
                    
                # Calculate loss
                loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
        
    def train(self) -> Dict[str, List[float]]:
        """Train the model.
        
        Returns:
            Dictionary containing training history
        """
        logger.info("Starting training...")
        start_time = time.time()
        
        epochs = self.config.get('epochs', 100)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Save checkpoint if best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    self.checkpoint_dir / "best_model.pth",
                    metadata={
                        'config': self.config,
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses
                    }
                )
                
            # Save regular checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth",
                    metadata={
                        'config': self.config,
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses
                    }
                )
                
            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        training_time = time.time() - start_time
        logger.info(f"Training completed in {format_time(training_time)}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time
        }
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        epoch, loss, metadata = load_checkpoint(
            self.model,
            self.optimizer,
            checkpoint_path
        )
        
        self.current_epoch = epoch
        self.best_val_loss = loss
        
        if 'train_losses' in metadata:
            self.train_losses = metadata['train_losses']
        if 'val_losses' in metadata:
            self.val_losses = metadata['val_losses']
            
        logger.info(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")


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
                if isinstance(self.model, nn.TransformerEncoder):
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
            if isinstance(self.model, nn.TransformerEncoder):
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
