import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import gc

# REVISION 1.2: Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class FeatureExtractor:
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, duration=4.0):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.target_length = int(np.ceil(self.duration * self.sample_rate / self.hop_length))
        self.freq_bins = self.n_fft // 2 + 1

    def extract_spectrogram(self, audio_path):
        try:
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # REVISION 2.1: Pad or truncate waveform
            target_samples = int(self.duration * self.sample_rate)
            if len(waveform) > target_samples:
                waveform = waveform[:target_samples]
            elif len(waveform) < target_samples:
                waveform = np.pad(waveform, (0, target_samples - len(waveform)), mode='constant')

            stft = librosa.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length)
            spectrogram = np.abs(stft) ** 2
            log_spectrogram = np.log1p(spectrogram)
            log_spectrogram = log_spectrogram.T

            current_time_steps = log_spectrogram.shape[0]
            if current_time_steps > self.target_length:
                log_spectrogram = log_spectrogram[:self.target_length, :]
            elif current_time_steps < self.target_length:
                pad_width = self.target_length - current_time_steps
                log_spectrogram = np.pad(log_spectrogram, ((0, pad_width), (0, 0)), mode='constant')

            if log_spectrogram.shape[1] != self.freq_bins:
                if log_spectrogram.shape[1] > self.freq_bins:
                    log_spectrogram = log_spectrogram[:, :self.freq_bins]
                else:
                    pad_width = self.freq_bins - log_spectrogram.shape[1]
                    log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
                    
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            log_spectrogram = np.zeros((self.target_length, self.freq_bins))
        
        return log_spectrogram

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

# REVISION 3.0: Updated data generator with stratified sampling
def data_generator(protocol_file, audio_folder_path, feature_extractor, batch_size=32, 
                   is_eval=False, sample_fraction=0.1):  # REVISED: Default 0.1 instead of 0.4
    data = pd.read_csv(protocol_file, sep='\s+', header=None, engine='python')
    
    if len(data.columns) == 5:
        data.columns = ['speaker_id', 'file_name', 'field1', 'system_id', 'label_text']
        data['label'] = data['label_text'].apply(lambda x: 0 if x == 'bonafide' else 1)
    elif len(data.columns) == 4 and is_eval:
        data.columns = ['speaker_id', 'file_name', 'field1', 'system_id']
        data['label'] = data['system_id'].apply(lambda x: 0 if x == 'bonafide' else 1)
    else:
        raise ValueError(f"Unexpected number of columns ({len(data.columns)}) in protocol file: {protocol_file}")
    
    if sample_fraction < 1.0:
        np.random.seed(42)
        
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
        
        print(f"Using {len(data)} samples ({sample_fraction*100:.1f}% of original dataset)")
    
    file_names = data['file_name'].values
    labels = data['label'].values
    
    def gen():
        for f, l in zip(file_names, labels):
            audio_path = os.path.join(audio_folder_path, f"{f}.flac")
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}. Skipping.")
                spec = np.zeros((feature_extractor.target_length, feature_extractor.freq_bins))
            else:
                spec = feature_extractor.extract_spectrogram(audio_path)
                
            spec = np.expand_dims(spec, axis=-1)
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

class MaxFeatureMap(layers.Layer):
    def __init__(self, **kwargs):
        super(MaxFeatureMap, self).__init__(**kwargs)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        channels = input_shape[-1]
        split = tf.split(inputs, num_or_size_splits=2, axis=-1)
        return tf.maximum(split[0], split[1])

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        if shape[-1] is not None:
            shape[-1] = shape[-1] // 2
        else:
            shape[-1] = None
        return tuple(shape)

# REVISION 4.0: Optimized residual block
def res_block(input_tensor, filters, stride=1):
    x = layers.Conv2D(filters * 2, kernel_size=3, strides=stride, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = MaxFeatureMap()(x)

    x = layers.Conv2D(filters * 2, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    shortcut = input_tensor
    input_channels_static = input_tensor.shape[-1]

    needs_projection = False
    if stride != 1:
        needs_projection = True
    if input_channels_static is not None and input_channels_static != (filters * 2):
        needs_projection = True

    if needs_projection:
        shortcut = layers.Conv2D(filters * 2, kernel_size=1, strides=stride, padding='same', use_bias=False)(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = MaxFeatureMap()(x)
    return x

def build_resmax(input_shape, num_classes=2):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = MaxFeatureMap()(x)

    x = res_block(x, filters=32, stride=1)
    x = res_block(x, filters=32, stride=1)

    x = res_block(x, filters=64, stride=2)
    x = res_block(x, filters=64, stride=1)

    x = res_block(x, filters=128, stride=2)
    x = res_block(x, filters=128, stride=1)

    x = res_block(x, filters=256, stride=2)
    x = res_block(x, filters=256, stride=1)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer = np.mean((fpr[eer_index], fnr[eer_index]))
    return eer * 100.0

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Dev Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Dev Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('resmax_training_history.png')
    print("Training history plot saved as resmax_training_history.png")
    plt.show()

# REVISION 5.0: Memory-optimized training function
def train_model(train_ds, dev_ds, eval_ds, input_shape, batch_size=16, num_epochs=5):
    model = build_resmax(input_shape=input_shape, num_classes=2)
    model.summary()
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    lr_reducer = callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=2, 
        verbose=1
    )
    early_stopper = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        verbose=1, 
        restore_best_weights=True
    )
    checkpoint = callbacks.ModelCheckpoint(
        "resmax_best_model.keras",
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1
    )
    
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=dev_ds,
        epochs=num_epochs,
        callbacks=[lr_reducer, early_stopper, checkpoint]
    )
    
    print("Evaluating on development set...")
    y_true_dev = []
    y_scores_dev = []
    
    for specs, labels in dev_ds:
        logits = model.predict(specs)
        probs = tf.nn.softmax(logits, axis=1).numpy()[:, 1]
        y_scores_dev.extend(probs)
        y_true_dev.extend(labels.numpy())
        
        del specs, labels, logits, probs
        gc.collect()
    
    auc_dev = roc_auc_score(y_true_dev, y_scores_dev)
    eer_dev = compute_eer(y_true_dev, y_scores_dev)
    print(f"Development Set AUC: {auc_dev:.4f}, EER: {eer_dev:.2f}%")
    
    print("Evaluating on evaluation set...")
    eval_loss, eval_acc = model.evaluate(eval_ds)
    print(f"Evaluation Set Loss: {eval_loss:.4f}, Accuracy: {eval_acc*100:.2f}%")
    
    y_true_eval = []
    y_scores_eval = []
    
    for specs, labels in eval_ds:
        logits = model.predict(specs)
        probs = tf.nn.softmax(logits, axis=1).numpy()[:, 1]
        y_scores_eval.extend(probs)
        y_true_eval.extend(labels.numpy())
        
        del specs, labels, logits, probs
        gc.collect()
    
    auc_eval = roc_auc_score(y_true_eval, y_scores_eval)
    eer_eval = compute_eer(y_true_eval, y_scores_eval)
    print(f"Evaluation Set AUC: {auc_eval:.4f}, EER: {eer_eval:.2f}%")
    
    model.save("resmax_final_model.keras")
    print("Model saved as resmax_final_model.keras")
    
    return model, history, (eval_loss, eval_acc, eer_dev, auc_dev, eer_eval, auc_eval)

def main():
    tf.random.set_seed(42)
    np.random.seed(42)
    
    base_dir = "/kaggle/input/asvpoof-2019-dataset/"
    paths = get_dataset_paths(base_dir)
    
    # REVISION 6.0: Reduced sample fraction for faster training
    sample_fraction = 0.1
    
    feature_extractor = FeatureExtractor()
    
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
    
    input_shape = (feature_extractor.target_length, feature_extractor.freq_bins, 1)
    print(f"Model input shape: {input_shape}")
    
    model, history, eval_results = train_model(
        train_ds,
        dev_ds,
        eval_ds,
        input_shape,
        batch_size=16,
        num_epochs=5
    )
    
    plot_training_history(history)
    
    eval_loss, eval_acc, dev_eer, dev_auc, eval_eer, eval_auc = eval_results
    print("\nFinal Results Summary:")
    print(f"Development Set: EER = {dev_eer:.2f}%, AUC = {dev_auc:.4f}")
    print(f"Evaluation Set: Loss = {eval_loss:.4f}, Accuracy = {eval_acc*100:.2f}%")
    print(f"Evaluation Set: EER = {eval_eer:.2f}%, AUC = {eval_auc:.4f}")

if __name__ == "__main__":
    main()