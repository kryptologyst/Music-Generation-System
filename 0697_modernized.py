#!/usr/bin/env python3
"""Modernized Music Generation System - Main entry point."""

import argparse
import logging
import yaml
from pathlib import Path
import torch
import numpy as np

from src.data.midi_dataset import MIDIDataset, MIDITokenizer
from src.models.music_generator import create_model
from src.train.trainer import MusicTrainer
from src.eval.evaluator import MusicEvaluator
from src.utils.utils import set_seed, get_device, create_directories

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main function for the modernized music generation system."""
    parser = argparse.ArgumentParser(description="Modernized Music Generation System")
    parser.add_argument("--mode", choices=["train", "generate", "evaluate"], default="generate")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data_dir", type=str, default="data/raw/midi_files")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    create_directories(args.output_dir, ["checkpoints", "logs", "assets"])
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    if args.mode == "train":
        train_model(config, args, device)
    elif args.mode == "generate":
        generate_music(config, args, device)
    elif args.mode == "evaluate":
        evaluate_model(config, args, device)


def train_model(config: dict, args: argparse.Namespace, device: torch.device):
    """Train the music generation model."""
    logger.info("Starting model training...")
    
    # Create dataset
    dataset = MIDIDataset(
        data_dir=args.data_dir,
        sequence_length=config['data']['sequence_length'],
        overlap=config['data']['overlap']
    )
    
    if len(dataset) == 0:
        logger.error("No MIDI files found. Please add MIDI files to the data directory.")
        return
    
    # Create data loaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Create model
    model = create_model(config['model'])
    model = model.to(device)
    
    # Create trainer
    trainer = MusicTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=device,
        checkpoint_dir=Path(args.output_dir) / "checkpoints"
    )
    
    # Train model
    training_history = trainer.train()
    logger.info("Training completed!")


def generate_music(config: dict, args: argparse.Namespace, device: torch.device):
    """Generate music using the trained model."""
    logger.info("Generating music...")
    
    # Load model
    if args.checkpoint is None:
        logger.error("Please provide a checkpoint path for generation.")
        return
    
    model = create_model(config['model'])
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create tokenizer
    tokenizer = MIDITokenizer()
    
    # Generate music
    with torch.no_grad():
        # Create seed sequence
        seed_length = config['generation']['seed_length']
        seed = torch.randint(4, 128, (1, seed_length)).to(device)
        
        # Generate sequence
        if hasattr(model, 'generate'):
            generated = model.generate(
                seed,
                max_length=config['generation']['max_length'],
                temperature=config['generation']['temperature'],
                top_k=config['generation']['top_k'],
                top_p=config['generation']['top_p']
            )
        else:
            # Simple generation fallback
            generated = seed.clone()
            for _ in range(config['generation']['max_length']):
                logits = model(generated)[:, -1, :]
                probs = torch.softmax(logits / config['generation']['temperature'], dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
        
        # Convert to MIDI events
        generated_sequence = generated[0].cpu().numpy()
        midi_events = tokenizer.decode(generated_sequence.tolist())
        
        # Save generated music
        output_path = Path(args.output_dir) / "generated_music.json"
        import json
        with open(output_path, 'w') as f:
            json.dump(midi_events, f, indent=2)
        
        logger.info(f"Generated music saved to {output_path}")


def evaluate_model(config: dict, args: argparse.Namespace, device: torch.device):
    """Evaluate the trained model."""
    logger.info("Evaluating model...")
    
    # Create dataset
    dataset = MIDIDataset(
        data_dir=args.data_dir,
        sequence_length=config['data']['sequence_length'],
        overlap=config['data']['overlap']
    )
    
    # Create data loader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Load model
    if args.checkpoint is None:
        logger.error("Please provide a checkpoint path for evaluation.")
        return
    
    model = create_model(config['model'])
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create evaluator
    evaluator = MusicEvaluator(
        model=model,
        test_loader=test_loader,
        tokenizer=dataset.tokenizer,
        device=device
    )
    
    # Evaluate model
    metrics = evaluator.evaluate()
    
    # Save results
    output_path = Path(args.output_dir) / "evaluation_results.json"
    import json
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    main()
