"""
Prediction script for image classification
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.data_loader import DataLoader


def load_model(model_path):
    """
    Load a trained model
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    return model


def predict_image(model, image_path, class_names, top_k=5):
    """
    Make prediction on a single image
    
    Args:
        model: Trained Keras model
        image_path: Path to image file
        class_names: List of class names
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary with predictions
    """
    # Load and preprocess image
    data_loader = DataLoader()
    img_array = data_loader.preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    
    # Get top k predictions
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    top_probs = predictions[0][top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    results = {
        'predictions': list(zip(top_classes, top_probs)),
        'top_class': top_classes[0],
        'top_confidence': float(top_probs[0])
    }
    
    return results


def predict_batch(model, image_paths, class_names, top_k=5):
    """
    Make predictions on multiple images
    
    Args:
        model: Trained Keras model
        image_paths: List of image paths
        class_names: List of class names
        top_k: Number of top predictions to return
        
    Returns:
        List of prediction dictionaries
    """
    results = []
    
    for image_path in image_paths:
        try:
            result = predict_image(model, image_path, class_names, top_k)
            result['image_path'] = image_path
            results.append(result)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            results.append({
                'image_path': image_path,
                'error': str(e)
            })
    
    return results


def visualize_prediction(image_path, predictions, save_path=None):
    """
    Visualize image with predictions
    
    Args:
        image_path: Path to image
        predictions: List of (class_name, probability) tuples
        save_path: Path to save visualization
    """
    # Load image
    img = Image.open(image_path)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('Input Image', fontsize=14, fontweight='bold')
    
    # Display predictions
    classes = [pred[0] for pred in predictions]
    probs = [pred[1] * 100 for pred in predictions]
    
    colors = ['green' if i == 0 else 'steelblue' for i in range(len(classes))]
    bars = ax2.barh(range(len(classes)), probs, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(classes)))
    ax2.set_yticklabels(classes)
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Top Predictions', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax2.text(prob + 1, bar.get_y() + bar.get_height()/2,
                f'{prob:.1f}%',
                va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def print_predictions(results):
    """
    Print prediction results in a formatted way
    
    Args:
        results: Prediction results dictionary
    """
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
    else:
        print(f"Top Prediction: {results['top_class']}")
        print(f"Confidence: {results['top_confidence']*100:.2f}%")
        print("\nAll Predictions:")
        print("-"*80)
        for i, (class_name, prob) in enumerate(results['predictions'], 1):
            print(f"{i}. {class_name:20s} - {prob*100:6.2f}%")
    
    print("="*80 + "\n")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Image Classification Prediction')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image file or directory')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (default: best_model.h5)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='Dataset name for class names')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to show')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize predictions')
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple images in directory')
    
    args = parser.parse_args()
    
    # Set model path
    model_path = args.model or config.MODEL_PATHS['best_model']
    
    # Load model
    model = load_model(model_path)
    
    # Get class names
    if args.dataset == 'cifar10':
        class_names = config.CIFAR10_CLASSES
    elif args.dataset == 'fashion_mnist':
        class_names = config.FASHION_MNIST_CLASSES
    else:
        class_names = config.CLASS_NAMES
    
    # Make predictions
    if args.batch and os.path.isdir(args.image):
        # Process directory
        image_paths = [
            os.path.join(args.image, f)
            for f in os.listdir(args.image)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        
        print(f"\nProcessing {len(image_paths)} images...")
        results_list = predict_batch(model, image_paths, class_names, args.top_k)
        
        # Print results
        for results in results_list:
            print(f"\nImage: {results['image_path']}")
            print_predictions(results)
    
    else:
        # Process single image
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return
        
        results = predict_image(model, args.image, class_names, args.top_k)
        print_predictions(results)
        
        # Visualize if requested
        if args.visualize:
            visualize_prediction(args.image, results['predictions'])
    
    print("Prediction completed!")


if __name__ == "__main__":
    main()

# Made with Bob
