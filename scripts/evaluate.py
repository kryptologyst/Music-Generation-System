#!/usr/bin/env python3
"""Evaluation script for music generation system."""

import argparse
import logging
import yaml
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data.midi_dataset import MIDIDataset, create_data_loaders
from src.models.music_generator import create_model
from src.eval.evaluator import MusicEvaluator
from src.utils.utils import set_seed, get_device, load_checkpoint
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
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate music generation model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
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
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
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
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config['model'])
    model = model.to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create evaluator
    evaluator = MusicEvaluator(
        model=model,
        test_loader=test_loader,
        tokenizer=dataset.tokenizer,
        device=device
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    test_metrics = evaluator.evaluate()
    
    # Save evaluation results
    eval_path = Path(args.output_dir) / "evaluation_results.json"
    with open(eval_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {eval_path}")
    
    # Print results
    logger.info("Evaluation Results:")
    logger.info("=" * 50)
    for metric_name, value in test_metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # Generate sample sequences
    logger.info(f"Generating {args.num_samples} sample sequences...")
    samples = evaluator.generate_samples(
        num_samples=args.num_samples,
        max_length=config['generation']['max_length'],
        temperature=config['generation']['temperature']
    )
    
    # Save samples
    samples_path = Path(args.output_dir) / "generated_samples.json"
    with open(samples_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    logger.info(f"Sample sequences saved to {samples_path}")
    
    # Generate detailed analysis
    logger.info("Generating detailed analysis...")
    
    # Analyze sample diversity
    all_tokens = []
    for sample in samples:
        all_tokens.extend(sample)
    
    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)
    diversity_score = unique_tokens / total_tokens if total_tokens > 0 else 0.0
    
    # Analyze note distribution
    note_tokens = [token for token in all_tokens if 4 <= token <= 127]
    if note_tokens:
        note_counts = {}
        for note in note_tokens:
            note_counts[note] = note_counts.get(note, 0) + 1
        
        # Find most common notes
        most_common_notes = sorted(note_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        analysis = {
            'diversity_score': diversity_score,
            'total_tokens': total_tokens,
            'unique_tokens': unique_tokens,
            'note_tokens': len(note_tokens),
            'most_common_notes': most_common_notes,
            'note_range': {
                'min': min(note_tokens) if note_tokens else 0,
                'max': max(note_tokens) if note_tokens else 0,
                'mean': sum(note_tokens) / len(note_tokens) if note_tokens else 0
            }
        }
        
        # Save analysis
        analysis_path = Path(args.output_dir) / "analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Analysis saved to {analysis_path}")
        
        # Print analysis
        logger.info("Sample Analysis:")
        logger.info("=" * 50)
        logger.info(f"Diversity Score: {diversity_score:.4f}")
        logger.info(f"Total Tokens: {total_tokens}")
        logger.info(f"Unique Tokens: {unique_tokens}")
        logger.info(f"Note Tokens: {len(note_tokens)}")
        logger.info(f"Note Range: {analysis['note_range']['min']} - {analysis['note_range']['max']}")
        logger.info(f"Average Note: {analysis['note_range']['mean']:.1f}")
        
        logger.info("Most Common Notes:")
        for note, count in most_common_notes:
            logger.info(f"  Note {note}: {count} occurrences")
    
    logger.info("Evaluation script completed successfully!")


if __name__ == "__main__":
    main()
