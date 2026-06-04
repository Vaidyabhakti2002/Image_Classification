# Deep Learning Image Classification - Quick Start Guide

Get started with the DL Image Classification project in minutes!

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

## 📊 Quick Training Examples

### Example 1: Train on CIFAR-10 (Built-in Dataset)

Train a custom CNN on CIFAR-10 dataset:

```bash
python scripts/train_custom_cnn.py --dataset cifar10 --epochs 20 --batch_size 32
```

### Example 2: Transfer Learning with ResNet50

```bash
python scripts/train_transfer_learning.py --model resnet50 --dataset cifar10 --epochs 15
```

### Example 3: Train on Fashion-MNIST

```bash
python scripts/train_custom_cnn.py --dataset fashion_mnist --epochs 25
```

## 🔮 Making Predictions

### Predict Single Image

```bash
python scripts/predict.py --image path/to/image.jpg --model models/best_model.h5 --visualize
```

### Predict Multiple Images

```bash
python scripts/predict.py --image path/to/images/ --model models/best_model.h5 --batch
```

## 📓 Using Jupyter Notebooks

### Start Jupyter

```bash
jupyter notebook
```

### Open Notebooks

1. `notebooks/01_data_exploration.ipynb` - Explore datasets
2. `notebooks/02_model_training.ipynb` - Train models interactively
3. `notebooks/03_model_evaluation.ipynb` - Evaluate and visualize results

## 🎯 Common Use Cases

### Use Case 1: Quick Test with Simple Model

```python
from utils.data_loader import DataLoader
from utils.model_builder import ModelBuilder

# Load CIFAR-10
loader = DataLoader('cifar10')
(x_train, y_train), (x_val, y_val), (x_test, y_test), classes = loader.load_builtin_dataset()

# Build and train simple model
builder = ModelBuilder(num_classes=10)
model = builder.build_simple_cnn()
model = builder.compile_model(model)

# Train
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10)
```

### Use Case 2: Transfer Learning

```python
from utils.model_builder import ModelBuilder, create_callbacks

# Build transfer learning model
builder = ModelBuilder(num_classes=10)
model = builder.build_transfer_learning_model('resnet50', trainable=False)
model = builder.compile_model(model)

# Train with callbacks
callbacks = create_callbacks('models/resnet50_model.h5')
history = model.fit(train_data, validation_data=val_data, epochs=20, callbacks=callbacks)
```

### Use Case 3: Custom Dataset

1. Organize your data:
```
data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       └── ...
├── validation/
│   └── ...
└── test/
    └── ...
```

2. Train:
```bash
python scripts/train_custom_cnn.py --dataset custom --epochs 30
```

## 📈 Monitor Training

### TensorBoard

```bash
tensorboard --logdir=logs
```

Then open http://localhost:6006 in your browser.

## 🎨 Visualization Examples

### Plot Training History

```python
from utils.visualization import plot_training_history

plot_training_history(history, save_path='training_plot.png')
```

### Confusion Matrix

```python
from utils.visualization import plot_confusion_matrix

plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png')
```

### Sample Predictions

```python
from utils.visualization import plot_sample_predictions

plot_sample_predictions(model, x_test, y_test, class_names, num_samples=16)
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Image settings
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Training settings
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Model settings
DROPOUT_RATE = 0.5
```

## 🐛 Troubleshooting

### Issue: Out of Memory

**Solution:** Reduce batch size in `config.py`:
```python
BATCH_SIZE = 16  # or 8
```

### Issue: Slow Training

**Solution:** Use GPU or reduce image size:
```python
IMG_HEIGHT = 128
IMG_WIDTH = 128
```

### Issue: Overfitting

**Solution:** Increase dropout or add data augmentation:
```python
DROPOUT_RATE = 0.6
# Enable augmentation in data loader
```

## 📚 Next Steps

1. **Experiment with architectures** - Try different CNN designs
2. **Tune hyperparameters** - Adjust learning rate, batch size, etc.
3. **Try different datasets** - Use your own images
4. **Deploy model** - Export to TensorFlow Lite or ONNX
5. **Advanced techniques** - Implement attention mechanisms, ensemble methods

## 🎓 Learning Path

1. Start with CIFAR-10 and simple CNN
2. Experiment with data augmentation
3. Try transfer learning with pre-trained models
4. Fine-tune models for better performance
5. Use your own custom dataset
6. Deploy and serve predictions

## 💡 Tips

- **Start small**: Use fewer epochs for quick experiments
- **Monitor metrics**: Watch for overfitting (val_loss increasing)
- **Save checkpoints**: Best models are automatically saved
- **Use callbacks**: Early stopping prevents wasted training time
- **Visualize**: Always plot training history and confusion matrix

## 🔗 Useful Commands

```bash
# List available models
ls models/

# Check TensorFlow GPU
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"

# View model summary
python -c "from tensorflow import keras; model = keras.models.load_model('models/best_model.h5'); model.summary()"

# Convert to TensorFlow Lite
python -c "import tensorflow as tf; converter = tf.lite.TFLiteConverter.from_keras_model(model); tflite_model = converter.convert()"
```

## 📞 Need Help?

- Check the main README.md for detailed documentation
- Review example notebooks in `notebooks/`
- Examine training scripts in `scripts/`
- Modify `config.py` for your needs

Happy Training! 🎉