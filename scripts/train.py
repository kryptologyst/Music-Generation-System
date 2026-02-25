#!/usr/bin/env python3
"""Training script for music generation system."""

import argparse
import logging
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data.midi_dataset import MIDIDataset, create_data_loaders
from src.models.music_generator import create_model
from src.train.trainer import MusicTrainer
from src.utils.utils import set_seed, get_device, create_directories
from src.metrics.metrics import MusicMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train music generation model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/midi_files",
        help="Directory containing MIDI files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    create_directories(args.output_dir, ["checkpoints", "logs", "assets"])
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create dataset
    logger.info("Loading dataset...")
    dataset = MIDIDataset(
        data_dir=args.data_dir,
        sequence_length=config['data']['sequence_length'],
        overlap=config['data']['overlap']
    )
    
    if len(dataset) == 0:
        logger.error("No MIDI files found in the dataset directory")
        return
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        batch_size=config['training']['batch_size'],
        train_split=0.8,
        val_split=0.1,
        num_workers=4
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} sequences")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config['model'])
    model = model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = MusicTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=device,
        checkpoint_dir=Path(args.output_dir) / "checkpoints"
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    logger.info("Starting training...")
    training_history = trainer.train()
    
    # Save training history
    import json
    history_path = Path(args.output_dir) / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info(f"Training completed. History saved to {history_path}")
    
    # Load best model for evaluation
    best_checkpoint = Path(args.output_dir) / "checkpoints" / "best_model.pth"
    if best_checkpoint.exists():
        logger.info("Loading best model for evaluation...")
        trainer.load_checkpoint(str(best_checkpoint))
        
        # Evaluate on test set
        from src.eval.evaluator import MusicEvaluator
        evaluator = MusicEvaluator(
            model=trainer.model,
            test_loader=test_loader,
            tokenizer=dataset.tokenizer,
            device=device
        )
        
        logger.info("Evaluating model...")
        test_metrics = evaluator.evaluate()
        
        # Save evaluation results
        eval_path = Path(args.output_dir) / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {eval_path}")
        
        # Generate sample sequences
        logger.info("Generating sample sequences...")
        samples = evaluator.generate_samples(
            num_samples=5,
            max_length=config['generation']['max_length'],
            temperature=config['generation']['temperature']
        )
        
        # Save samples
        samples_path = Path(args.output_dir) / "generated_samples.json"
        with open(samples_path, 'w') as f:
            json.dump(samples, f, indent=2)
        
        logger.info(f"Sample sequences saved to {samples_path}")
    
    logger.info("Training script completed successfully!")


if __name__ == "__main__":
    main()
