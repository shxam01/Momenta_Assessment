import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def compute_eer(y_true, y_scores):
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer = np.mean((fpr[eer_index], fnr[eer_index]))
    return eer * 100.0

def evaluate_model(model, dataset):
    
    
    y_true = []
    y_scores = []
    
    for specs, labels in dataset:
        logits = model.predict(specs)
        probs = tf.nn.softmax(logits, axis=1).numpy()[:, 1]  # Get probability of spoof class
        y_scores.extend(probs)
        y_true.extend(labels.numpy())
    
   
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    
    y_pred = (y_scores > 0.5).astype(int)
    

    accuracy = np.mean(y_pred == y_true)
    

    eer = compute_eer(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    
    results = {
        'accuracy': accuracy,
        'auc': auc,
        'eer': eer,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'y_true': y_true,
        'y_scores': y_scores,
        'y_pred': y_pred
    }
    
    # results
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"AUC: {auc:.4f}")
    print(f"EER: {eer:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    
    return results
