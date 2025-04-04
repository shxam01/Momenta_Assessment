import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from data_utils import load_protocol_file, get_dataset_paths

def analyze_dataset_balance(protocol_dir):
    
    splits = ['train', 'dev', 'eval']
    file_names = {
        'train': 'ASVspoof2019.LA.cm.train.trn.txt',
        'dev': 'ASVspoof2019.LA.cm.dev.trl.txt',
        'eval': 'ASVspoof2019.LA.cm.eval.trl.txt'
    }
    
    plt.figure(figsize=(12, 5))
    
    for i, split in enumerate(splits):
        protocol_file = os.path.join(protocol_dir, file_names[split])
        is_eval = split == 'eval'
        
        data = load_protocol_file(protocol_file, is_eval=is_eval, sample_fraction=1.0)
        if data.empty:
            print(f"Skipping {split} split - could not load data")
            continue
            

        bonafide_count = sum(data['label'] == 0)
        spoof_count = sum(data['label'] == 1)
        total = len(data)
        
        print(f"{split.capitalize()} split:")
        print(f"  Total samples: {total}")
        print(f"  Bonafide samples: {bonafide_count} ({bonafide_count/total*100:.1f}%)")
        print(f"  Spoof samples: {spoof_count} ({spoof_count/total*100:.1f}%)")
        print()
        
        # Plot class distribution
        plt.subplot(1, 3, i+1)
        plt.bar(['Bonafide', 'Spoof'], [bonafide_count, spoof_count])
        plt.title(f"{split.capitalize()} Split")
        plt.ylabel("Number of samples")
        plt.grid(axis='y', alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.show()

def visualize_audio_samples(audio_base_dir, protocol_dir, num_samples=3):
    
    #  train protocol file 
    train_protocol = os.path.join(protocol_dir, 'ASVspoof2019.LA.cm.train.trn.txt')
    train_audio_dir = os.path.join(audio_base_dir, 'ASVspoof2019_LA_train/flac')
    
    data = load_protocol_file(train_protocol, sample_fraction=1.0)
    if data.empty:
        print("Could not load training data for visualization")
        return
        
    #  random samples from each class
    bonafide_samples = data[data['label'] == 0].sample(num_samples)
    spoof_samples = data[data['label'] == 1].sample(num_samples)
    
    samples = pd.concat([bonafide_samples, spoof_samples])
    
    plt.figure(figsize=(15, 10))
    for i, (_, row) in enumerate(samples.iterrows()):
        audio_path = os.path.join(train_audio_dir, f"{row['file_name']}.flac")
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            continue
            
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Plot waveform
        plt.subplot(num_samples * 2, 2, i*2 + 1)
        plt.title(f"{'Bonafide' if row['label'] == 0 else 'Spoof'} - Waveform")
        librosa.display.waveshow(audio, sr=sr)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        
        # Plot spectrogram
        plt.subplot(num_samples * 2, 2, i*2 + 2)
        plt.title(f"{'Bonafide' if row['label'] == 0 else 'Spoof'} - Spectrogram")
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig('audio_samples.png')
    plt.show()

def duration_analysis(audio_base_dir, protocol_dir):
    
    
    train_protocol = os.path.join(protocol_dir, 'ASVspoof2019.LA.cm.train.trn.txt')
    train_audio_dir = os.path.join(audio_base_dir, 'ASVspoof2019_LA_train/flac')
    
    data = load_protocol_file(train_protocol, sample_fraction=0.1)  # Use only 10% for speed
    if data.empty:
        print("Could not load training data for duration analysis")
        return
        
    
    durations = []
    for _, row in data.iterrows():
        audio_path = os.path.join(train_audio_dir, f"{row['file_name']}.flac")
        if os.path.exists(audio_path):
            audio, sr = librosa.load(audio_path, sr=None)
            duration = len(audio) / sr
            durations.append(duration)
    
   
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=30)
    plt.title('Audio File Duration Distribution')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    plt.savefig('duration_distribution.png')
    plt.show()
    
    print(f"Duration Statistics:")
    print(f"  Min: {min(durations):.2f} seconds")
    print(f"  Max: {max(durations):.2f} seconds")
    print(f"  Mean: {np.mean(durations):.2f} seconds")
    print(f"  Median: {np.median(durations):.2f} seconds")

def run_eda(base_dir):
   
    paths = get_dataset_paths(base_dir)
    
    print("Analyzing dataset balance...")
    analyze_dataset_balance(paths['protocol']['train'].rsplit('/', 1)[0])
    
    print("\nVisualizing audio samples...")
    visualize_audio_samples(base_dir, paths['protocol']['train'].rsplit('/', 1)[0])
    
    print("\nAnalyzing audio durations...")
    duration_analysis(base_dir, paths['protocol']['train'].rsplit('/', 1)[0])
    
    print("\nEDA complete. Plots saved to disk.")
