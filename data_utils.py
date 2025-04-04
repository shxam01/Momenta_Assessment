import os
import pandas as pd
import numpy as np

def load_protocol_file(protocol_file, is_eval=False, sample_fraction=0.4, random_seed=42):
   
    try:
        data = pd.read_csv(protocol_file, sep='\s+', header=None, engine='python')
        
        if len(data.columns) == 5:
            data.columns = ['speaker_id', 'file_name', 'field1', 'system_id', 'label_text']
            data['label'] = data['label_text'].apply(lambda x: 0 if x == 'bonafide' else 1)
        elif len(data.columns) == 4 and is_eval:
            data.columns = ['speaker_id', 'file_name', 'field1', 'system_id']
            data['label'] = data['system_id'].apply(lambda x: 0 if x == 'bonafide' else 1)
        else:
            raise ValueError(f"Unexpected number of columns ({len(data.columns)}) in protocol file: {protocol_file}")
        
        # Sample the data
        if sample_fraction < 1.0:
            # Stratified sampling to maintain class distribution
            np.random.seed(random_seed)
            
            
            bonafide_idx = data[data['label'] == 0].index
            spoof_idx = data[data['label'] == 1].index
            
            
            sampled_bonafide = np.random.choice(bonafide_idx, 
                                             size=int(len(bonafide_idx) * sample_fraction),
                                             replace=False)
            sampled_spoof = np.random.choice(spoof_idx, 
                                          size=int(len(spoof_idx) * sample_fraction),
                                          replace=False)
            
            
            sampled_indices = np.concatenate([sampled_bonafide, sampled_spoof])
            

            data = data.loc[sampled_indices].reset_index(drop=True)
            
            print(f"Loaded {len(data)} samples ({sample_fraction*100:.1f}% of original data)")
        
        return data
    except Exception as e:
        print(f"Error reading protocol file {protocol_file}: {e}")
        return pd.DataFrame()

def get_dataset_paths(base_dir):
    
    protocol_dir = os.path.join(base_dir, "LA/LA/ASVspoof2019_LA_cm_protocols/")
    
    paths = {
        'protocol': {
            'train': os.path.join(protocol_dir, 'ASVspoof2019.LA.cm.train.trn.txt'),
            'dev': os.path.join(protocol_dir, 'ASVspoof2019.LA.cm.dev.trl.txt'),
            'eval': os.path.join(protocol_dir, 'ASVspoof2019.LA.cm.eval.trl.txt')
        },
        'audio': {
            'train': os.path.join(base_dir, 'LA/LA/ASVspoof2019_LA_train/flac'),
            'dev': os.path.join(base_dir, 'LA/LA/ASVspoof2019_LA_dev/flac'),
            'eval': os.path.join(base_dir, 'LA/LA/ASVspoof2019_LA_eval/flac')
        }
    }
    
    
    for key, path in paths['protocol'].items():
        if not os.path.exists(path):
            print(f"Warning: Protocol file not found: {path}")
    
    for key, path in paths['audio'].items():
        if not os.path.exists(path):
            print(f"Warning: Audio directory not found: {path}")
    
    return paths
