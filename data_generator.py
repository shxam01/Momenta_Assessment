import tensorflow as tf
import os
import numpy as np
import pandas as pd
from feature_extraction import FeatureExtractor

def data_generator(protocol_file, audio_folder_path, feature_extractor, batch_size=32, 
                   is_eval=False, sample_fraction=0.4):
    
    # Load protocol data
    data = pandas.read_csv(protocol_file, sep='\s+', header=None, engine='python')
    
    if len(data.columns) == 5:
        data.columns = ['speaker_id', 'file_name', 'field1', 'system_id', 'label_text']
        data['label'] = data['label_text'].apply(lambda x: 0 if x == 'bonafide' else 1)
    elif len(data.columns) == 4 and is_eval:
        data.columns = ['speaker_id', 'file_name', 'field1', 'system_id']
        data['label'] = data['system_id'].apply(lambda x: 0 if x == 'bonafide' else 1)
    else:
        raise ValueError(f"Unexpected number of columns ({len(data.columns)}) in protocol file: {protocol_file}")
    
    # Data index
    if sample_fraction < 1.0:
        # Stratified sampling to maintain class distribution
        np.random.seed(42)  # For reproducibility
        
        # Bonafide and spoof index
        bonafide_idx = data[data['label'] == 0].index
        spoof_idx = data[data['label'] == 1].index
        
        
        sampled_bonafide = np.random.choice(bonafide_idx, 
                                         size=int(len(bonafide_idx) * sample_fraction),
                                         replace=False)
        sampled_spoof = np.random.choice(spoof_idx, 
                                      size=int(len(spoof_idx) * sample_fraction),
                                      replace=False)
        
        # Combined sampled indices
        sampled_indices = np.concatenate([sampled_bonafide, sampled_spoof])
        
        # DATA SAMPLE
        data = data.loc[sampled_indices].reset_index(drop=True)
        
        print(f"Using {len(data)} samples ({sample_fraction*100:.1f}% of original dataset)")
    
    file_names = data['file_name'].values
    labels = data['label'].values
    
    def gen():
        for f, l in zip(file_names, labels):
            audio_path = os.path.join(audio_folder_path, f"{f}.flac")
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}. Skipping.")
                # Use zeros if file not found
                spec = np.zeros((feature_extractor.target_length, feature_extractor.freq_bins))
            else:
                spec = feature_extractor.extract_spectrogram(audio_path)
                
            spec = np.expand_dims(spec, axis=-1)  # Add channel dimension
            yield spec.astype(np.float32), np.int32(l)

    output_signature = (
        tf.TensorSpec(shape=(feature_extractor.target_length, feature_extractor.freq_bins, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if not is_eval:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
