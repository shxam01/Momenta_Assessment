import numpy as np
import librosa
import os

class FeatureExtractor:
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, duration=4.0):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration  # seconds
        self.target_length = int(np.ceil(self.duration * self.sample_rate / self.hop_length))
        self.freq_bins = self.n_fft // 2 + 1

    def extract_spectrogram(self, audio_path):
        
        try:
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
            
           
            target_samples = int(self.duration * self.sample_rate)
            if len(waveform) > target_samples:
                waveform = waveform[:target_samples]
            elif len(waveform) < target_samples:
                waveform = np.pad(waveform, (0, target_samples - len(waveform)), mode='constant')

            stft = librosa.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length)
            spectrogram = np.abs(stft) ** 2
            log_spectrogram = np.log1p(spectrogram)
            log_spectrogram = log_spectrogram.T  # shape: (time, freq)

            
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
        
        return log_spectrogram  # shape: (time, freq)
