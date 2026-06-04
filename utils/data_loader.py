"""
Data loading and preprocessing utilities for image classification
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10, fashion_mnist, mnist
import config


class DataLoader:
    """Class to handle data loading and preprocessing"""
    
    def __init__(self, dataset_name='custom'):
        """
        Initialize DataLoader
        
        Args:
            dataset_name: Name of dataset ('cifar10', 'fashion_mnist', 'mnist', 'custom')
        """
        self.dataset_name = dataset_name
        self.img_height = config.IMG_HEIGHT
        self.img_width = config.IMG_WIDTH
        self.batch_size = config.BATCH_SIZE
        
    def load_builtin_dataset(self):
        """
        Load built-in datasets (CIFAR-10, Fashion-MNIST, MNIST)
        
        Returns:
            Tuple of (x_train, y_train), (x_val, y_val), (x_test, y_test)
        """
        if self.dataset_name == 'cifar10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            class_names = config.CIFAR10_CLASSES
            
        elif self.dataset_name == 'fashion_mnist':
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            # Add channel dimension for grayscale images
            x_train = np.expand_dims(x_train, axis=-1)
            x_test = np.expand_dims(x_test, axis=-1)
            class_names = config.FASHION_MNIST_CLASSES
            
        elif self.dataset_name == 'mnist':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            # Add channel dimension for grayscale images
            x_train = np.expand_dims(x_train, axis=-1)
            x_test = np.expand_dims(x_test, axis=-1)
            class_names = [str(i) for i in range(10)]
            
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Split training data into train and validation
        val_split = int(len(x_train) * config.VALIDATION_SPLIT)
        x_val = x_train[:val_split]
        y_val = y_train[:val_split]
        x_train = x_train[val_split:]
        y_train = y_train[val_split:]
        
        # Convert labels to categorical
        num_classes = len(class_names)
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_val = keras.utils.to_categorical(y_val, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        
        print(f"Dataset: {self.dataset_name}")
        print(f"Training samples: {len(x_train)}")
        print(f"Validation samples: {len(x_val)}")
        print(f"Test samples: {len(x_test)}")
        print(f"Number of classes: {num_classes}")
        print(f"Image shape: {x_train.shape[1:]}")
        
        return (x_train, y_train), (x_val, y_val), (x_test, y_test), class_names
    
    def create_data_generators(self, augment_train=True):
        """
        Create data generators for custom datasets
        
        Args:
            augment_train: Whether to apply data augmentation to training data
            
        Returns:
            Tuple of (train_generator, val_generator, test_generator)
        """
        # Training data generator with augmentation
        if augment_train:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=config.AUGMENTATION_CONFIG['rotation_range'],
                width_shift_range=config.AUGMENTATION_CONFIG['width_shift_range'],
                height_shift_range=config.AUGMENTATION_CONFIG['height_shift_range'],
                horizontal_flip=config.AUGMENTATION_CONFIG['horizontal_flip'],
                vertical_flip=config.AUGMENTATION_CONFIG['vertical_flip'],
                zoom_range=config.AUGMENTATION_CONFIG['zoom_range'],
                shear_range=config.AUGMENTATION_CONFIG['shear_range'],
                brightness_range=config.AUGMENTATION_CONFIG['brightness_range'],
                fill_mode=config.AUGMENTATION_CONFIG['fill_mode']
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        # Validation and test data generators (no augmentation)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            config.TRAIN_DIR,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=config.RANDOM_SEED
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            config.VAL_DIR,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = None
        if os.path.exists(config.TEST_DIR) and os.listdir(config.TEST_DIR):
            test_generator = val_test_datagen.flow_from_directory(
                config.TEST_DIR,
                target_size=(self.img_height, self.img_width),
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )
        
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        if test_generator:
            print(f"Test samples: {test_generator.samples}")
        print(f"Number of classes: {train_generator.num_classes}")
        
        return train_generator, val_generator, test_generator
    
    def create_tf_dataset(self, x_data, y_data, is_training=True):
        """
        Create TensorFlow dataset from numpy arrays
        
        Args:
            x_data: Input images
            y_data: Labels
            is_training: Whether this is training data (for shuffling)
            
        Returns:
            tf.data.Dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(x_data))
            dataset = dataset.map(self._augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    @tf.function
    def _augment_image(self, image, label):
        """
        Apply data augmentation to a single image
        
        Args:
            image: Input image
            label: Image label
            
        Returns:
            Augmented image and label
        """
        # Random flip
        image = tf.image.random_flip_left_right(image)
        
        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.2)
        
        # Random contrast
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        
        # Ensure values are in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        img = keras.preprocessing.image.load_img(
            image_path,
            target_size=(self.img_height, self.img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def get_class_weights(self, y_train):
        """
        Calculate class weights for imbalanced datasets
        
        Args:
            y_train: Training labels
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        # Convert one-hot to class indices
        if len(y_train.shape) > 1:
            y_train = np.argmax(y_train, axis=1)
        
        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        
        class_weights = dict(zip(classes, weights))
        
        print("Class weights:", class_weights)
        return class_weights


def download_sample_dataset():
    """
    Download and prepare a sample dataset (CIFAR-10)
    """
    print("Downloading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Image shape: {x_train.shape[1:]}")
    print("Dataset downloaded successfully!")
    
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    # Test data loader
    print("Testing DataLoader...")
    
    # Test with CIFAR-10
    loader = DataLoader(dataset_name='cifar10')
    (x_train, y_train), (x_val, y_val), (x_test, y_test), class_names = loader.load_builtin_dataset()
    
    print("\nDataLoader test completed successfully!")

# Made with Bob
