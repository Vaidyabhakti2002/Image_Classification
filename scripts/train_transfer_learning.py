"""
Training script for transfer learning models
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


def train_transfer_learning(base_model='resnet50', dataset_name='cifar10', 
                            epochs=None, batch_size=None, fine_tune=False):
    """
    Train a transfer learning model
    
    Args:
        base_model: Pre-trained model to use ('vgg16', 'resnet50', 'mobilenet', 'efficientnet')
        dataset_name: Name of dataset to use
        epochs: Number of training epochs
        batch_size: Batch size for training
        fine_tune: Whether to fine-tune the base model
    """
    print("="*80)
    print(f"Training Transfer Learning Model: {base_model.upper()}")
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
        
        # Resize images if needed for transfer learning
        if x_train.shape[1] != config.IMG_HEIGHT or x_train.shape[2] != config.IMG_WIDTH:
            print(f"Resizing images from {x_train.shape[1:3]} to {(config.IMG_HEIGHT, config.IMG_WIDTH)}...")
            x_train = tf.image.resize(x_train, (config.IMG_HEIGHT, config.IMG_WIDTH)).numpy()
            x_val = tf.image.resize(x_val, (config.IMG_HEIGHT, config.IMG_WIDTH)).numpy()
            x_test = tf.image.resize(x_test, (config.IMG_HEIGHT, config.IMG_WIDTH)).numpy()
        
        # Convert grayscale to RGB if needed
        if x_train.shape[-1] == 1:
            print("Converting grayscale to RGB...")
            x_train = np.repeat(x_train, 3, axis=-1)
            x_val = np.repeat(x_val, 3, axis=-1)
            x_test = np.repeat(x_test, 3, axis=-1)
        
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
    print(f"\n2. Building {base_model} transfer learning model...")
    num_classes = len(class_names)
    model_builder = ModelBuilder(num_classes=num_classes)
    model = model_builder.build_transfer_learning_model(
        base_model_name=base_model,
        trainable=fine_tune
    )
    model = model_builder.compile_model(model, optimizer='adam')
    model_builder.print_model_summary(model)
    
    # Create callbacks
    print("\n3. Setting up callbacks...")
    model_path = os.path.join(config.MODELS_DIR, f'{base_model}_transfer.h5')
    callbacks = create_callbacks(model_path=model_path)
    
    # Train model
    print("\n4. Training model...")
    print(f"Base model: {base_model}")
    print(f"Fine-tuning: {fine_tune}")
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
    
    # Fine-tune if requested
    if fine_tune:
        print("\n5. Fine-tuning model...")
        # Unfreeze base model
        for layer in model.layers:
            if hasattr(layer, 'trainable'):
                layer.trainable = True
        
        # Recompile with lower learning rate
        model = model_builder.compile_model(model, optimizer='adam', learning_rate=config.LEARNING_RATE / 10)
        
        # Continue training
        fine_tune_epochs = epochs // 2
        history_fine = model.fit(
            train_dataset,
            epochs=fine_tune_epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        for key in history.history.keys():
            history.history[key].extend(history_fine.history[key])
    
    # Evaluate model
    print("\n6. Evaluating model...")
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
    print("\n7. Generating visualizations...")
    plot_training_history(history, save_path=f'training_history_{base_model}.png')
    
    # Generate predictions for confusion matrix
    if dataset_name in ['cifar10', 'fashion_mnist', 'mnist']:
        y_pred = model.predict(test_dataset)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        plot_confusion_matrix(
            y_true_classes,
            y_pred_classes,
            class_names,
            save_path=f'confusion_matrix_{base_model}.png'
        )
    
    # Save final model
    final_model_path = os.path.join(config.MODELS_DIR, f'{base_model}_final.h5')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)
    
    return model, history


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Transfer Learning Model')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['vgg16', 'resnet50', 'mobilenet', 'efficientnet'],
                       help='Pre-trained model to use')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='Dataset name (cifar10, fashion_mnist, mnist, custom)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--fine_tune', action='store_true',
                       help='Fine-tune the base model')
    
    args = parser.parse_args()
    
    # Train model
    model, history = train_transfer_learning(
        base_model=args.model,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fine_tune=args.fine_tune
    )
    
    print("\nTraining script completed!")


if __name__ == "__main__":
    main()

# Made with Bob
