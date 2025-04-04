import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
import librosa.display

from feature_extraction import FeatureExtractor

def predict_audio(model, audio_path):
    """
    Make a prediction on a single audio file
    
    Args:
        model: Trained model
        audio_path: Path to audio file
        
    Returns:
        prediction: Dictionary with prediction results
    """
    # Extract features
    feature_extractor = FeatureExtractor()
    spec = feature_extractor.extract_spectrogram(audio_path)
    spec = np.expand_dims(spec, axis=0)  # Add batch dimension
    spec = np.expand_dims(spec, axis=-1)  # Add channel dimension
    
    # Make prediction
    logits = model.predict(spec)
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    
    # Get class prediction
    prediction_idx = np.argmax(probs)
    prediction = "Bonafide" if prediction_idx == 0 else "Spoof"
    
    # Return results
    return {
        "prediction": prediction,
        "bonafide_probability": float(probs[0]),
        "spoof_probability": float(probs[1])
    }

def visualize_prediction(audio_path, prediction):
    """
    Visualize audio waveform and spectrogram with prediction
    
    Args:
        audio_path: Path to audio file
        prediction: Dictionary with prediction results
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform - Prediction: {prediction['prediction']} "
              f"(Bonafide: {prediction['bonafide_probability']:.2%}, "
              f"Spoof: {prediction['spoof_probability']:.2%})")
    
    # Plot spectrogram
    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.show()

def main():
    # Load model
    model = tf.keras.models.load_model(
        'resmax_final_model.h5',
        custom_objects={'MaxFeatureMap': MaxFeatureMap}
    )
    
    # Audio file path - replace with an actual file path
    audio_path = "/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_1000001.flac"
    
    # Make prediction
    prediction = predict_audio(model, audio_path)
    
    # Print results
    print(f"File: {os.path.basename(audio_path)}")
    print(f"Prediction: {prediction['prediction']}")
    print(f"Bonafide probability: {prediction['bonafide_probability']:.4f}")
    print(f"Spoof probability: {prediction['spoof_probability']:.4f}")
    
    # Visualize
    visualize_prediction(audio_path, prediction)

if __name__ == "__main__":
    # Import MaxFeatureMap for model loading
    from model import MaxFeatureMap
    main()