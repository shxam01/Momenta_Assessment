import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_utils import get_dataset_paths, load_protocol_file
from feature_extraction import FeatureExtractor
from data_generator import data_generator
from model import build_resmax
from train import train_model
from evaluate import evaluate_model
from visualize import plot_training_history, plot_roc_curve, plot_confusion_matrix

def main():
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Define dataset paths
    base_dir = "/kaggle/input/asvpoof-2019-dataset/"
    paths = get_dataset_paths(base_dir)
    
    # Set the fraction of data to use (35-40% of total)
    sample_fraction = 0.4
    
    # Create feature extractor
    feature_extractor = FeatureExtractor()
    
    # Create datasets with reduced size
    print("Creating datasets...")
    train_ds = data_generator(
        paths['protocol']['train'],
        paths['audio']['train'],
        feature_extractor,
        batch_size=16,
        sample_fraction=sample_fraction
    )
    
    dev_ds = data_generator(
        paths['protocol']['dev'],
        paths['audio']['dev'],
        feature_extractor,
        batch_size=16,
        sample_fraction=sample_fraction
    )
    
    eval_ds = data_generator(
        paths['protocol']['eval'],
        paths['audio']['eval'],
        feature_extractor,
        batch_size=16,
        sample_fraction=sample_fraction,
        is_eval=True
    )
    
    # Input shape for the model
    input_shape = (feature_extractor.target_length, feature_extractor.freq_bins, 1)
    print(f"Model input shape: {input_shape}")
    
    # Train the model
    model, history, eval_results = train_model(
        train_ds,
        dev_ds,
        eval_ds,
        input_shape,
        batch_size=16,
        num_epochs=5
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on all sets
    print("\nFinal Evaluation:")
    
    # Extract results from training
    eval_loss, eval_acc, dev_eer, dev_auc, eval_eer, eval_auc = eval_results
    
    # Print summary
    print("\nFinal Results Summary:")
    print(f"Development Set: EER = {dev_eer:.2f}%, AUC = {dev_auc:.4f}")
    print(f"Evaluation Set: Loss = {eval_loss:.4f}, Accuracy = {eval_acc*100:.2f}%")
    print(f"Evaluation Set: EER = {eval_eer:.2f}%, AUC = {eval_auc:.4f}")

if __name__ == "__main__":
    main()