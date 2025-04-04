import tensorflow as tf
from tensorflow.keras import optimizers, losses, callbacks
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import gc  # Garbage collector for memory management

def compute_eer(y_true, y_scores):
    """Compute Equal Error Rate"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer = np.mean((fpr[eer_index], fnr[eer_index]))
    return eer * 100.0

def train_model(train_ds, dev_ds, eval_ds, input_shape, batch_size=16, num_epochs=5):
    """
    Train and evaluate the ResMax model
    
    Args:
        train_ds: Training dataset 
        dev_ds: Development dataset
        eval_ds: Evaluation dataset
        input_shape: Model input shape
        batch_size: Batch size
        num_epochs: Number of training epochs
        
    Returns:
        model: Trained model
        history: Training history
        eval_results: Evaluation results tuple
    """
    # Build model
    from model import build_resmax
    model = build_resmax(input_shape=input_shape, num_classes=2)
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Callbacks
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
        "resmax_best_model.h5", 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1
    )
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=dev_ds,
        epochs=num_epochs,
        callbacks=[lr_reducer, early_stopper, checkpoint]
    )
    
    # Evaluate on dev set (EER/AUC)
    print("Evaluating on development set...")
    y_true_dev = []
    y_scores_dev = []
    
    for specs, labels in dev_ds:
        logits = model.predict(specs)
        probs = tf.nn.softmax(logits, axis=1).numpy()[:, 1]  # Get probability of spoof class
        y_scores_dev.extend(probs)
        y_true_dev.extend(labels.numpy())
        
        # Free memory
        del specs, labels, logits, probs
        gc.collect()
    
    auc_dev = roc_auc_score(y_true_dev, y_scores_dev)
    eer_dev = compute_eer(y_true_dev, y_scores_dev)
    print(f"Development Set AUC: {auc_dev:.4f}, EER: {eer_dev:.2f}%")
    
    # Evaluate on evaluation set (loss/accuracy)
    print("Evaluating on evaluation set...")
    eval_loss, eval_acc = model.evaluate(eval_ds)
    print(f"Evaluation Set Loss: {eval_loss:.4f}, Accuracy: {eval_acc*100:.2f}%")
    
    # Calculate EER/AUC on evaluation set if labels are available
    y_true_eval = []
    y_scores_eval = []
    
    for specs, labels in eval_ds:
        logits = model.predict(specs)
        probs = tf.nn.softmax(logits, axis=1).numpy()[:, 1]
        y_scores_eval.extend(probs)
        y_true_eval.extend(labels.numpy())
        
        # Free memory
        del specs, labels, logits, probs
        gc.collect()
    
    auc_eval = roc_auc_score(y_true_eval, y_scores_eval)
    eer_eval = compute_eer(y_true_eval, y_scores_eval)
    print(f"Evaluation Set AUC: {auc_eval:.4f}, EER: {eer_eval:.2f}%")
    
    # Save final model
    model.save("resmax_final_model.h5")
    print("Model saved as resmax_final_model.h5")
    
    return model, history, (eval_loss, eval_acc, eer_dev, auc_dev, eer_eval, auc_eval)