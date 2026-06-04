"""
Configuration file for Deep Learning Image Classification Project
"""

import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Image Parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Model Parameters
NUM_CLASSES = 10  # Update based on your dataset
DROPOUT_RATE = 0.5

# Data Augmentation Parameters
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': False,
    'zoom_range': 0.2,
    'shear_range': 0.15,
    'brightness_range': [0.8, 1.2],
    'fill_mode': 'nearest'
}

# Transfer Learning Models
PRETRAINED_MODELS = {
    'vgg16': {
        'input_shape': (224, 224, 3),
        'weights': 'imagenet',
        'include_top': False
    },
    'resnet50': {
        'input_shape': (224, 224, 3),
        'weights': 'imagenet',
        'include_top': False
    },
    'mobilenet': {
        'input_shape': (224, 224, 3),
        'weights': 'imagenet',
        'include_top': False
    },
    'efficientnet': {
        'input_shape': (224, 224, 3),
        'weights': 'imagenet',
        'include_top': False
    }
}

# Optimizer Configuration
OPTIMIZER_CONFIG = {
    'name': 'adam',
    'learning_rate': LEARNING_RATE,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-07
}

# Learning Rate Scheduler
LR_SCHEDULER_CONFIG = {
    'factor': 0.5,
    'patience': 5,
    'min_lr': 1e-7,
    'verbose': 1
}

# Early Stopping Configuration
EARLY_STOPPING_CONFIG = {
    'monitor': 'val_loss',
    'patience': 10,
    'restore_best_weights': True,
    'verbose': 1
}

# Model Checkpoint Configuration
CHECKPOINT_CONFIG = {
    'monitor': 'val_accuracy',
    'save_best_only': True,
    'mode': 'max',
    'verbose': 1
}

# TensorBoard Configuration
TENSORBOARD_CONFIG = {
    'log_dir': LOGS_DIR,
    'histogram_freq': 1,
    'write_graph': True,
    'write_images': True,
    'update_freq': 'epoch'
}

# Class Names (Update based on your dataset)
CLASS_NAMES = [
    'class_0', 'class_1', 'class_2', 'class_3', 'class_4',
    'class_5', 'class_6', 'class_7', 'class_8', 'class_9'
]

# CIFAR-10 Class Names (Example)
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Fashion-MNIST Class Names (Example)
FASHION_MNIST_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Random Seed for Reproducibility
RANDOM_SEED = 42

# GPU Configuration
GPU_CONFIG = {
    'allow_growth': True,
    'memory_limit': None  # Set to specific MB if needed, e.g., 4096
}

# Evaluation Metrics
METRICS = ['accuracy', 'precision', 'recall', 'auc']

# Prediction Configuration
PREDICTION_CONFIG = {
    'top_k': 5,  # Number of top predictions to show
    'confidence_threshold': 0.5
}

# Dataset Configuration
DATASET_CONFIG = {
    'name': 'custom',  # Options: 'cifar10', 'fashion_mnist', 'custom'
    'color_mode': 'rgb',  # Options: 'rgb', 'grayscale'
    'class_mode': 'categorical',  # Options: 'categorical', 'binary'
    'shuffle': True,
    'seed': RANDOM_SEED
}

# Model Save Paths
MODEL_PATHS = {
    'custom_cnn': os.path.join(MODELS_DIR, 'custom_cnn.h5'),
    'transfer_learning': os.path.join(MODELS_DIR, 'transfer_learning.h5'),
    'best_model': os.path.join(MODELS_DIR, 'best_model.h5'),
    'checkpoint': os.path.join(MODELS_DIR, 'checkpoint.h5')
}

# Visualization Configuration
VIZ_CONFIG = {
    'figsize': (12, 8),
    'dpi': 100,
    'style': 'seaborn-v0_8-darkgrid',
    'save_format': 'png'
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
}

# Mixed Precision Training
USE_MIXED_PRECISION = False

# Multi-GPU Training
USE_MULTI_GPU = False
NUM_GPUS = 1

# Model Export Configuration
EXPORT_CONFIG = {
    'onnx': True,
    'tflite': True,
    'saved_model': True
}

print(f"Configuration loaded successfully!")
print(f"Base Directory: {BASE_DIR}")
print(f"Image Size: {IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Number of Classes: {NUM_CLASSES}")

# Made with Bob
