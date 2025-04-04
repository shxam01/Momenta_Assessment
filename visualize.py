import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix

def plot_training_history(history):
    """Plot training history metrics"""
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

def plot_roc_curve(y_true, y_scores):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    auc = np.trapz(tpr, fpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Bonafide', 'Spoof']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def visualize_spectrograms(audio_files, feature_extractor, predictions=None, num_samples=3):
    """Visualize spectrograms of a few audio samples"""
    plt.figure(figsize=(15, num_samples * 4))
    
    for i, audio_path in enumerate(audio_files[:num_samples]):
        # Extract spectrogram
        spec = feature_extractor.extract_spectrogram(audio_path)
        
        # Plot spectrogram
        plt.subplot(num_samples, 1, i+1)
        title = f"File: {os.path.basename(audio_path)}"
        if predictions is not None:
            title += f", Prediction: {predictions[i]}"
        plt.title(title)
        plt.imshow(spec.T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.xlabel('Time Frames')
        plt.ylabel('Frequency Bins')
    
    plt.tight_layout()
    plt.savefig('spectrograms.png')
    plt.show()