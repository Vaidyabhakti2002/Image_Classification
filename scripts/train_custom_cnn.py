"""
Training script for custom CNN model
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.data_loader import DataLoader
from utils.model_builder import ModelBuilder, create_callbacks
from utils.visualization import plot_training_history, plot_confusion_matrix


def train_custom_cnn(dataset_name='cifar10', epochs=None, batch_size=None):
    """
    Train a custom CNN model
    
    Args:
        dataset_name: Name of dataset to use
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("="*80)
    print("Training Custom CNN Model")
    print("="*80)
    
    # Set parameters
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    
    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)
    
    # Load data
    print("\n1. Loading dataset...")
    data_loader = DataLoader(dataset_name=dataset_name)
    
    if dataset_name in ['cifar10', 'fashion_mnist', 'mnist']:
        (x_train, y_train), (x_val, y_val), (x_test, y_test), class_names = data_loader.load_builtin_dataset()
        
        # Create TensorFlow datasets
        train_dataset = data_loader.create_tf_dataset(x_train, y_train, is_training=True)
        val_dataset = data_loader.create_tf_dataset(x_val, y_val, is_training=False)
        test_dataset = data_loader.create_tf_dataset(x_test, y_test, is_training=False)
        
        steps_per_epoch = len(x_train) // batch_size
        validation_steps = len(x_val) // batch_size
    else:
        train_gen, val_gen, test_gen = data_loader.create_data_generators(augment_train=True)
        train_dataset = train_gen
        val_dataset = val_gen
        test_dataset = test_gen
        class_names = list(train_gen.class_indices.keys())
        
        steps_per_epoch = train_gen.samples // batch_size
        validation_steps = val_gen.samples // batch_size
    
    # Build model
    print("\n2. Building model...")
    num_classes = len(class_names)
    model_builder = ModelBuilder(num_classes=num_classes)
    model = model_builder.build_custom_cnn()
    model = model_builder.compile_model(model, optimizer='adam')
    model_builder.print_model_summary(model)
    
    # Create callbacks
    print("\n3. Setting up callbacks...")
    model_path = config.MODEL_PATHS['custom_cnn']
    callbacks = create_callbacks(model_path=model_path)
    
    # Train model
    print("\n4. Training model...")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print("-"*80)
    
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\n5. Evaluating model...")
    test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(
        test_dataset,
        verbose=1
    )
    
    print("\n" + "="*80)
    print("Test Results:")
    print("="*80)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print("="*80)
    
    # Plot training history
    print("\n6. Generating visualizations...")
    plot_training_history(history, save_path='training_history_custom_cnn.png')
    
    # Generate predictions for confusion matrix
    if dataset_name in ['cifar10', 'fashion_mnist', 'mnist']:
        y_pred = model.predict(test_dataset)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        plot_confusion_matrix(
            y_true_classes,
            y_pred_classes,
            class_names,
            save_path='confusion_matrix_custom_cnn.png'
        )
    
    # Save final model
    final_model_path = os.path.join(config.MODELS_DIR, 'custom_cnn_final.h5')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)
    
    return model, history


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Custom CNN Model')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='Dataset name (cifar10, fashion_mnist, mnist, custom)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    
    args = parser.parse_args()
    
    # Train model
    model, history = train_custom_cnn(
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print("\nTraining script completed!")


if __name__ == "__main__":
    main()

# Made with Bob
