"""
Model building utilities for image classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNetV2, EfficientNetB0
)
import config


class ModelBuilder:
    """Class to build various CNN architectures"""
    
    def __init__(self, input_shape=None, num_classes=None):
        """
        Initialize ModelBuilder
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape or (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
        self.num_classes = num_classes or config.NUM_CLASSES
        
    def build_custom_cnn(self, name='custom_cnn'):
        """
        Build a custom CNN architecture
        
        Args:
            name: Model name
            
        Returns:
            Keras model
        """
        model = models.Sequential(name=name)
        
        # First Convolutional Block
        model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=self.input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Second Convolutional Block
        model.add(layers.Conv2D(64, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Third Convolutional Block
        model.add(layers.Conv2D(128, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Fourth Convolutional Block
        model.add(layers.Conv2D(256, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.GlobalAveragePooling2D())
        
        # Dense Layers
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(config.DROPOUT_RATE))
        
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.3))
        
        # Output Layer
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def build_simple_cnn(self, name='simple_cnn'):
        """
        Build a simple CNN for quick testing
        
        Args:
            name: Model name
            
        Returns:
            Keras model
        """
        model = models.Sequential(name=name)
        
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def build_transfer_learning_model(self, base_model_name='resnet50', trainable=False):
        """
        Build a transfer learning model using pre-trained weights
        
        Args:
            base_model_name: Name of pre-trained model ('vgg16', 'resnet50', 'mobilenet', 'efficientnet')
            trainable: Whether to make base model trainable
            
        Returns:
            Keras model
        """
        # Load pre-trained base model
        if base_model_name.lower() == 'vgg16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name.lower() == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name.lower() == 'mobilenet':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name.lower() == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unknown base model: {base_model_name}")
        
        # Freeze base model layers
        base_model.trainable = trainable
        
        # Build complete model
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(config.DROPOUT_RATE)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs, name=f'{base_model_name}_transfer')
        
        return model
    
    def build_residual_block_model(self, name='resnet_custom'):
        """
        Build a custom model with residual blocks
        
        Args:
            name: Model name
            
        Returns:
            Keras model
        """
        def residual_block(x, filters, kernel_size=3, stride=1):
            """Create a residual block"""
            shortcut = x
            
            # First conv layer
            x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            # Second conv layer
            x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Adjust shortcut if needed
            if stride != 1 or shortcut.shape[-1] != filters:
                shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            
            # Add shortcut
            x = layers.Add()([x, shortcut])
            x = layers.Activation('relu')(x)
            
            return x
        
        # Build model
        inputs = keras.Input(shape=self.input_shape)
        
        # Initial conv layer
        x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual blocks
        x = residual_block(x, 64)
        x = residual_block(x, 64)
        
        x = residual_block(x, 128, stride=2)
        x = residual_block(x, 128)
        
        x = residual_block(x, 256, stride=2)
        x = residual_block(x, 256)
        
        # Global pooling and output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(config.DROPOUT_RATE)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs, name=name)
        
        return model
    
    def compile_model(self, model, optimizer='adam', learning_rate=None):
        """
        Compile a model with specified optimizer and metrics
        
        Args:
            model: Keras model to compile
            optimizer: Optimizer name or instance
            learning_rate: Learning rate (if None, uses config value)
            
        Returns:
            Compiled model
        """
        lr = learning_rate or config.LEARNING_RATE
        
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=lr)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=lr)
        else:
            opt = optimizer
        
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def print_model_summary(self, model):
        """
        Print model architecture summary
        
        Args:
            model: Keras model
        """
        print("\n" + "="*80)
        print(f"Model: {model.name}")
        print("="*80)
        model.summary()
        print("="*80)
        
        # Count parameters
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        print("="*80 + "\n")


def create_callbacks(model_path=None):
    """
    Create training callbacks
    
    Args:
        model_path: Path to save best model
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Model checkpoint
    if model_path:
        checkpoint = keras.callbacks.ModelCheckpoint(
            model_path,
            monitor=config.CHECKPOINT_CONFIG['monitor'],
            save_best_only=config.CHECKPOINT_CONFIG['save_best_only'],
            mode=config.CHECKPOINT_CONFIG['mode'],
            verbose=config.CHECKPOINT_CONFIG['verbose']
        )
        callbacks.append(checkpoint)
    
    # Early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor=config.EARLY_STOPPING_CONFIG['monitor'],
        patience=config.EARLY_STOPPING_CONFIG['patience'],
        restore_best_weights=config.EARLY_STOPPING_CONFIG['restore_best_weights'],
        verbose=config.EARLY_STOPPING_CONFIG['verbose']
    )
    callbacks.append(early_stop)
    
    # Learning rate scheduler
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.LR_SCHEDULER_CONFIG['factor'],
        patience=config.LR_SCHEDULER_CONFIG['patience'],
        min_lr=config.LR_SCHEDULER_CONFIG['min_lr'],
        verbose=config.LR_SCHEDULER_CONFIG['verbose']
    )
    callbacks.append(lr_scheduler)
    
    # TensorBoard
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=config.TENSORBOARD_CONFIG['log_dir'],
        histogram_freq=config.TENSORBOARD_CONFIG['histogram_freq'],
        write_graph=config.TENSORBOARD_CONFIG['write_graph'],
        write_images=config.TENSORBOARD_CONFIG['write_images'],
        update_freq=config.TENSORBOARD_CONFIG['update_freq']
    )
    callbacks.append(tensorboard)
    
    return callbacks


if __name__ == "__main__":
    # Test model builder
    print("Testing ModelBuilder...")
    
    builder = ModelBuilder(num_classes=10)
    
    # Build and display custom CNN
    model = builder.build_custom_cnn()
    builder.print_model_summary(model)
    
    print("\nModelBuilder test completed successfully!")

# Made with Bob
