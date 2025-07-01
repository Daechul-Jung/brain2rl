#!/usr/bin/env python
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.pipelines.eeg_to_rl_pipeline import EEGTokenizationPipeline, EEGActionVisualization
from models.rl.eeg_rl_integration import EEGRLIntegration
from models.analysis.action_analysis import ActionAnalyzer

def parse_args():
    parser = argparse.ArgumentParser(description="Run EEG to RL pipeline with action visualization")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default=os.path.join("data", "train"),
                        help="Path to the directory containing EEG data")
    parser.add_argument("--subjects", type=str, default="1,2,3",
                        help="Comma-separated list of subject IDs to use")
    parser.add_argument("--series", type=str, default=None,
                        help="Comma-separated list of series IDs to use (optional)")
    
    # Model parameters
    parser.add_argument("--tokenizer_path", type=str, 
                        default=os.path.join("models", "tokenization", "best_eeg_tokenizer.pth"),
                        help="Path to the pretrained tokenizer model")
    parser.add_argument("--classifier_path", type=str, 
                        default="best_eeg_action_classifier.pth",
                        help="Path to the pretrained action classifier model")
    
    # Pipeline parameters
    parser.add_argument("--window_size", type=int, default=500,
                        help="Window size for EEG data segmentation")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Overlap between consecutive windows (0-1)")
    
    # RL parameters
    parser.add_argument("--env_name", type=str, default="Pendulum-v0",
                        help="Name of the gym environment to use")
    parser.add_argument("--train_rl", action="store_true",
                        help="Whether to train the RL agent")
    parser.add_argument("--timesteps", type=int, default=10000,
                        help="Number of timesteps to train the RL agent")
    parser.add_argument("--token_usage_prob", type=float, default=0.7,
                        help="Probability of using EEG tokens during RL training")
    
    # Analysis parameters
    parser.add_argument("--analyze_actions", action="store_true",
                        help="Whether to analyze action patterns in the data")
    parser.add_argument("--visualize_subject", type=str, default=None,
                        help="Subject ID to visualize (e.g., 'subj1')")
    parser.add_argument("--visualize_series", type=str, default=None,
                        help="Series ID to visualize (e.g., 'series1')")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    
    return parser.parse_args()

def create_directories(args):
    """Create necessary directories for outputs"""
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "tokenization"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "rl"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "analysis"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "visualization"), exist_ok=True)

def parse_list(list_str):
    """Parse comma-separated list strings"""
    if list_str is None:
        return None
    return [int(x) for x in list_str.split(",")]

def run_tokenization_pipeline(args):
    """Run the EEG tokenization pipeline"""
    print("\n===== Running EEG Tokenization Pipeline =====")
    
    subject_ids = parse_list(args.subjects)
    series_ids = parse_list(args.series)
    
    pipeline = EEGTokenizationPipeline(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer_path,
        classifier_path=args.classifier_path,
        window_size=args.window_size,
        overlap=args.overlap
    )
    
    dataset = pipeline.load_data(subject_ids=subject_ids, series_ids=series_ids)
    
    print("Running tokenization pipeline...")
    tokens, actions, labels = pipeline.run_pipeline(batch_size=32, max_batches=10)
    
    print(f"Generated {tokens.shape[0]} token sequences")
    print(f"Token shape: {tokens.shape}")
    print(f"Action shape: {actions.shape}")
    
    np.save(os.path.join(args.output_dir, "tokenization", "sample_tokens.npy"), tokens[:10])
    np.save(os.path.join(args.output_dir, "tokenization", "sample_actions.npy"), actions[:10])
    
    pipeline.visualize_token_action_relationship(tokens, actions, n_samples=3)
    
    for i in range(1, 4):
        if os.path.exists(f"token_action_sample_{i}.png"):
            os.rename(
                f"token_action_sample_{i}.png", 
                os.path.join(args.output_dir, "tokenization", f"token_action_sample_{i}.png")
            )
    
    return pipeline, tokens, actions

def run_rl_integration(args, tokens=None, actions=None):
    """Run the RL integration with EEG tokens"""
    if not args.train_rl:
        print("\nSkipping RL training (--train_rl not specified)")
        return None
    
    print("\n===== Running RL Integration with EEG Tokens =====")
    
    subject_ids = parse_list(args.subjects)
    series_ids = parse_list(args.series)
      
    integration = EEGRLIntegration(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer_path,
        classifier_path=args.classifier_path,
        env_name=args.env_name,
        window_size=args.window_size,
        overlap=args.overlap
    )
    
    # Prepare token buffer
    if tokens is not None and actions is not None:
        # Use pre-computed tokens and actions
        integration.token_buffer.add(tokens, actions)
        print(f"Added {len(tokens)} pre-computed token sequences to buffer")
    else:
        # Generate new tokens and actions
        tokens, actions = integration.prepare_token_buffer(
            subject_ids=subject_ids,
            series_ids=series_ids,
            max_batches=10
        )
    
    # Initialize the agent
    agent = integration.initialize_agent()
    
    # Train the agent
    print(f"Training RL agent for {args.timesteps} timesteps...")
    token_usage = integration.train_with_eeg_guidance(
        total_timesteps=args.timesteps,
        token_usage_prob=args.token_usage_prob,
        save_path=os.path.join(args.output_dir, "rl", "eeg_guided_rl_model")
    )
    
    # Evaluate the agent
    print("Evaluating the RL agent...")
    avg_reward, rewards = integration.evaluate_with_eeg_tokens(num_episodes=5)
    
    # Move generated plots to output directory
    if os.path.exists("token_usage_history.png"):
        os.rename(
            "token_usage_history.png", 
            os.path.join(args.output_dir, "rl", "token_usage_history.png")
        )
    
    if os.path.exists("eeg_guided_rewards.png"):
        os.rename(
            "eeg_guided_rewards.png", 
            os.path.join(args.output_dir, "rl", "eeg_guided_rewards.png")
        )
    
    print(f"RL integration complete! Average reward: {avg_reward:.2f}")
    
    return integration

def run_action_analysis(args):
    """Run action pattern analysis"""
    if not args.analyze_actions:
        print("\nSkipping action analysis (--analyze_actions not specified)")
        return None
    
    print("\n===== Running Action Pattern Analysis =====")
    
    # Parse subject IDs
    subject_ids = parse_list(args.subjects)
    
    # Create analyzer
    analyzer = ActionAnalyzer(
        data_dir=args.data_dir,
        classifier_path=args.classifier_path,
        window_size=args.window_size,
        overlap=args.overlap
    )
    
    # Load data
    analyzer.load_data(subject_ids=subject_ids)
    
    # Analyze action patterns
    print("Analyzing action patterns...")
    results = analyzer.analyze_action_patterns()
    analyzer.plot_action_analysis(results)
    
    # Identify common action sequences
    print("Identifying common action sequences...")
    top_sequences, all_sequences = analyzer.identify_action_sequences(min_sequence_length=5)
    analyzer.plot_common_sequences(top_sequences)
    
    # Evaluate classifier if available
    if os.path.exists(args.classifier_path):
        print("Evaluating classifier...")
        eval_results = analyzer.evaluate_classifier(test_subject_ids=[subject_ids[-1]])
    
    # Move generated plots to output directory
    for filename in ["action_counts.png", "action_co_occurrence.png", "action_transitions.png", 
                     "common_action_sequences.png", "classifier_confusion_matrices.png"]:
        if os.path.exists(filename):
            os.rename(
                filename, 
                os.path.join(args.output_dir, "analysis", filename)
            )
    
    return analyzer

def visualize_subject(args, analyzer=None):
    """Visualize actions for a specific subject"""
    if args.visualize_subject is None or args.visualize_series is None:
        print("\nSkipping subject visualization (subject or series not specified)")
        return
    
    print(f"\n===== Visualizing Subject {args.visualize_subject}, Series {args.visualize_series} =====")
    
    if analyzer is None:
        # Create and initialize analyzer
        analyzer = ActionAnalyzer(
            data_dir=args.data_dir,
            classifier_path=args.classifier_path,
            window_size=args.window_size,
            overlap=args.overlap
        )
        
        # Parse subject IDs
        subject_ids = [int(args.visualize_subject.replace("subj", ""))]
        analyzer.load_data(subject_ids=subject_ids)
    
    # Visualize the subject
    window_data, window_events = analyzer.visualize_subject_actions(
        subject_id=args.visualize_subject,
        series_id=args.visualize_series,
        window_start=0,
        window_length=1000
    )
    
    # Move generated plots to output directory
    for filename in ["eeg_actions_visualization.png", "action_heatmap.png", "action_prediction_comparison.png"]:
        if os.path.exists(filename):
            os.rename(
                filename, 
                os.path.join(args.output_dir, "visualization", filename)
            )
    
    return window_data, window_events

def main():
    args = parse_args()
    
    create_directories(args)
    
    print("=== EEG to RL Pipeline Configuration ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Subjects: {args.subjects}")
    print(f"Series: {args.series or 'All'}")
    print(f"Tokenizer path: {args.tokenizer_path}")
    print(f"Classifier path: {args.classifier_path}")
    print(f"Window size: {args.window_size}")
    print(f"Output directory: {args.output_dir}")
    print("=====================================")
    
    pipeline, tokens, actions = run_tokenization_pipeline(args)
    
    integration = run_rl_integration(args, tokens, actions)
    
    analyzer = run_action_analysis(args)
    
    visualize_subject(args, analyzer)
    
    print("\n===== Pipeline Complete =====")
    print(f"Results saved to {args.output_dir}")
    
    print("\nSummary:")
    print(f"- Tokenized {tokens.shape[0]} EEG segments")
    
    if args.train_rl:
        print(f"- Trained RL agent for {args.timesteps} timesteps with token guidance")
    
    if args.analyze_actions:
        print("- Analyzed action patterns in the EEG data")
    
    if args.visualize_subject is not None:
        print(f"- Visualized actions for subject {args.visualize_subject}, series {args.visualize_series}")
    
    print("\nTo visualize the results, check the output directory for generated plots.")


if __name__ == "__main__":
    main() 