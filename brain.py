import os
import sys
import torch
import numpy as np
import argparse

# Check CUDA availability for GPU acceleration
print("CUDA available devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
    device = torch.device("cuda:0")
else:
    print("CUDA not available, using CPU")
    device = torch.device("cpu")

print("Device:", device)

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    parser = argparse.ArgumentParser(description="Brain to Agent's Actions")
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'simulate'],
                        help='Operation mode: train, evaluate, or run simulation')
    parser.add_argument('--model', type=str, default='classification',
                        choices=['classification', 'tokenization', 'rl'],
                        help='Model type to train or evaluate')
    parser.add_argument('--data', type=str, default='eeg',
                        choices=['eeg', 'fmri'],
                        help='Data type to use')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.mode == 'train':
        if args.model == 'classification':
            from models.classification.action_classifier import train_classifier
            train_classifier(args)
        elif args.model == 'tokenization':
            from models.tokenization.brain_tokenizer import train_tokenizer
            train_tokenizer(args)
        elif args.model == 'rl':
            from models.rl.brain_rl import train_rl
            train_rl(args)
    
    elif args.mode == 'eval':
        if args.model == 'classification':
            from models.classification.action_classifier import evaluate_classifier
            evaluate_classifier(args)
        elif args.model == 'tokenization':
            from models.tokenization.brain_tokenizer import evaluate_tokenizer
            evaluate_tokenizer(args)
        elif args.model == 'rl':
            from models.rl.brain_rl import evaluate_rl
            evaluate_rl(args)
    
    elif args.mode == 'simulate':
        from simulation.ros_interface import run_simulation
        run_simulation(args)

if __name__ == "__main__":
    main()