"""
Visualization utilities for model training and evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import config


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy)
    
    Args:
        history: Keras training history object
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_classification_report(y_true, y_pred, class_names, save_path=None):
    """
    Generate and display classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the report
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    print("\n" + "="*80)
    print("Classification Report")
    print("="*80)
    print(report)
    print("="*80)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("="*80 + "\n")
            f.write(report)
            f.write("="*80 + "\n")
        print(f"Classification report saved to: {save_path}")


def plot_sample_predictions(model, x_test, y_test, class_names, num_samples=16, save_path=None):
    """
    Plot sample predictions with true and predicted labels
    
    Args:
        model: Trained model
        x_test: Test images
        y_test: Test labels
        class_names: List of class names
        num_samples: Number of samples to display
        save_path: Path to save the plot
    """
    # Get predictions
    predictions = model.predict(x_test[:num_samples])
    pred_classes = np.argmax(predictions, axis=1)
    
    # Convert one-hot encoded labels to class indices
    if len(y_test.shape) > 1:
        true_classes = np.argmax(y_test[:num_samples], axis=1)
    else:
        true_classes = y_test[:num_samples]
    
    # Create subplot grid
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx in range(num_samples):
        img = x_test[idx]
        
        # Denormalize if needed
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        # Handle grayscale images
        if img.shape[-1] == 1:
            img = img.squeeze()
            axes[idx].imshow(img, cmap='gray')
        else:
            axes[idx].imshow(img)
        
        true_label = class_names[true_classes[idx]]
        pred_label = class_names[pred_classes[idx]]
        confidence = predictions[idx][pred_classes[idx]] * 100
        
        # Color code: green for correct, red for incorrect
        color = 'green' if true_classes[idx] == pred_classes[idx] else 'red'
        
        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%',
                           fontsize=10, color=color, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide extra subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions saved to: {save_path}")
    
    plt.show()


def plot_metrics_comparison(histories, model_names, save_path=None):
    """
    Compare training metrics across multiple models
    
    Args:
        histories: List of training history objects
        model_names: List of model names
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot training accuracy
    for history, name in zip(histories, model_names):
        axes[0, 0].plot(history.history['accuracy'], label=name, linewidth=2)
    axes[0, 0].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot validation accuracy
    for history, name in zip(histories, model_names):
        axes[0, 1].plot(history.history['val_accuracy'], label=name, linewidth=2)
    axes[0, 1].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot training loss
    for history, name in zip(histories, model_names):
        axes[1, 0].plot(history.history['loss'], label=name, linewidth=2)
    axes[1, 0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot validation loss
    for history, name in zip(histories, model_names):
        axes[1, 1].plot(history.history['val_loss'], label=name, linewidth=2)
    axes[1, 1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to: {save_path}")
    
    plt.show()


def plot_class_distribution(y_data, class_names, title='Class Distribution', save_path=None):
    """
    Plot class distribution in dataset
    
    Args:
        y_data: Labels (one-hot or class indices)
        class_names: List of class names
        title: Plot title
        save_path: Path to save the plot
    """
    # Convert one-hot to class indices if needed
    if len(y_data.shape) > 1:
        y_data = np.argmax(y_data, axis=1)
    
    # Count samples per class
    unique, counts = np.unique(y_data, return_counts=True)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(unique)), counts, color='steelblue', alpha=0.8)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to: {save_path}")
    
    plt.show()


def plot_learning_rate_schedule(history, save_path=None):
    """
    Plot learning rate schedule during training
    
    Args:
        history: Keras training history object
        save_path: Path to save the plot
    """
    if 'lr' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['lr'], linewidth=2, color='steelblue')
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning rate schedule saved to: {save_path}")
        
        plt.show()
    else:
        print("Learning rate not tracked in history")


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")

# Made with Bob
