# Deep Learning Image Classification Project

A comprehensive deep learning project for image classification using Convolutional Neural Networks (CNN) with TensorFlow/Keras.

## 🎯 Project Overview

This project demonstrates image classification using deep learning techniques including:
- Custom CNN architecture
- Transfer learning with pre-trained models (VGG16, ResNet50, MobileNet)
- Data augmentation techniques
- Model evaluation and visualization
- Real-time prediction capabilities

## 📁 Project Structure

```
dl_image_classification/
├── data/                      # Dataset directory
│   ├── train/                # Training images
│   ├── validation/           # Validation images
│   └── test/                 # Test images
├── models/                    # Saved models
│   ├── custom_cnn.h5
│   ├── transfer_learning.h5
│   └── best_model.h5
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
├── scripts/                   # Python scripts
│   ├── train_custom_cnn.py
│   ├── train_transfer_learning.py
│   ├── evaluate_model.py
│   └── predict.py
├── utils/                     # Utility functions
│   ├── data_loader.py
│   ├── model_builder.py
│   ├── visualization.py
│   └── preprocessing.py
├── requirements.txt           # Project dependencies
├── config.py                  # Configuration settings
└── README.md                  # This file
```

## 🚀 Features

### 1. Custom CNN Architecture
- Multiple convolutional layers with batch normalization
- Dropout for regularization
- Global average pooling
- Dense layers with activation functions

### 2. Transfer Learning
- Pre-trained models: VGG16, ResNet50, MobileNetV2
- Fine-tuning capabilities
- Feature extraction mode

### 3. Data Augmentation
- Random rotation, flip, zoom
- Brightness and contrast adjustment
- Normalization and preprocessing

### 4. Model Evaluation
- Accuracy, precision, recall, F1-score
- Confusion matrix visualization
- ROC curves and AUC scores
- Training history plots

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow
- OpenCV

## 🔧 Installation

1. Clone the repository or navigate to the project directory
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset

This project supports various image classification datasets:
- CIFAR-10 (10 classes)
- MNIST (handwritten digits)
- Fashion-MNIST (clothing items)
- Custom datasets (organize in train/val/test folders)

### Dataset Structure
```
data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       ├── img1.jpg
│       └── img2.jpg
├── validation/
│   └── ...
└── test/
    └── ...
```

## 🎓 Usage

### 1. Train Custom CNN Model
```bash
python scripts/train_custom_cnn.py --epochs 50 --batch_size 32
```

### 2. Train with Transfer Learning
```bash
python scripts/train_transfer_learning.py --model resnet50 --epochs 30
```

### 3. Evaluate Model
```bash
python scripts/evaluate_model.py --model_path models/best_model.h5
```

### 4. Make Predictions
```bash
python scripts/predict.py --image path/to/image.jpg --model models/best_model.h5
```

### 5. Using Jupyter Notebooks
```bash
jupyter notebook notebooks/02_model_training.ipynb
```

## 🏗️ Model Architectures

### Custom CNN
- Conv2D (32 filters) → BatchNorm → ReLU → MaxPool
- Conv2D (64 filters) → BatchNorm → ReLU → MaxPool
- Conv2D (128 filters) → BatchNorm → ReLU → MaxPool
- GlobalAveragePooling2D
- Dense (256) → Dropout (0.5)
- Dense (num_classes) → Softmax

### Transfer Learning
- Pre-trained base (VGG16/ResNet50/MobileNet)
- GlobalAveragePooling2D
- Dense (512) → Dropout (0.5)
- Dense (num_classes) → Softmax

## 📈 Performance Metrics

The project tracks and visualizes:
- Training/Validation Accuracy
- Training/Validation Loss
- Confusion Matrix
- Classification Report
- ROC Curves
- Precision-Recall Curves

## 🎨 Visualization Examples

- Training history plots
- Sample predictions with confidence scores
- Misclassified images analysis
- Feature map visualizations
- Grad-CAM heatmaps

## 🔍 Key Techniques

1. **Data Preprocessing**
   - Image resizing and normalization
   - One-hot encoding for labels
   - Train/validation/test split

2. **Data Augmentation**
   - Prevents overfitting
   - Increases dataset diversity
   - Improves model generalization

3. **Regularization**
   - Dropout layers
   - L2 regularization
   - Batch normalization

4. **Optimization**
   - Adam optimizer
   - Learning rate scheduling
   - Early stopping
   - Model checkpointing

## 📝 Configuration

Edit `config.py` to customize:
- Image dimensions
- Batch size
- Number of epochs
- Learning rate
- Model architecture
- Data augmentation parameters

## 🎯 Results

Expected performance (depends on dataset):
- Custom CNN: 85-90% accuracy
- Transfer Learning: 90-95% accuracy
- Training time: 10-30 minutes (GPU recommended)

## 🚀 Advanced Features

- Mixed precision training
- Multi-GPU support
- TensorBoard integration
- Model quantization
- ONNX export for deployment

## 📚 Learning Resources

- TensorFlow Documentation
- Keras API Reference
- Deep Learning Specialization (Coursera)
- CS231n: Convolutional Neural Networks

## 🤝 Contributing

Feel free to enhance this project by:
- Adding new model architectures
- Implementing advanced techniques
- Improving documentation
- Adding more datasets

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Created as a comprehensive deep learning project for image classification.

## 🙏 Acknowledgments

- TensorFlow/Keras team
- Pre-trained model providers
- Open-source community