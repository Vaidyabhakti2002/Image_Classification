# How to Run Predictions - Step by Step Guide

## Current Status
✅ Python 3.11.9 installed
✅ Virtual environment created
✅ TensorFlow and dependencies installed
⏳ Model training in progress (CIFAR-10 dataset)

## Wait for Training to Complete

The training is currently running in Terminal 1. You'll know it's complete when you see:
- "Epoch 5/5" finishing
- "Model saved to models/custom_cnn.h5"
- The command prompt returns

**Estimated time:** 5-15 minutes total (dataset download + training)

## Once Training Completes

### Option 1: Test on CIFAR-10 Images

The model will be trained on CIFAR-10, so you can test it on CIFAR-10 test images:

```powershell
cd C:\Users\BhaktiVaidya\Box\Bob_Agent\dl_image_classification
.\venv\Scripts\python.exe scripts\predict.py --model models/custom_cnn.h5 --dataset cifar10
```

This will automatically load test images from CIFAR-10 and show predictions.

### Option 2: Test on Your Own Image

If you have an image file (airplane, car, bird, cat, etc.):

```powershell
.\venv\Scripts\python.exe scripts\predict.py --image "C:\path\to\your\image.jpg" --model models/custom_cnn.h5 --visualize
```

Replace `C:\path\to\your\image.jpg` with your actual image path.

### Option 3: Create a Quick Test

Let me create a simple test script for you:

```python
# test_prediction.py
from tensorflow import keras
import numpy as np

# Load the trained model
model = keras.models.load_model('models/custom_cnn.h5')

# Load CIFAR-10 test data
from tensorflow.keras.datasets import cifar10
(_, _), (x_test, y_test) = cifar10.load_data()

# Normalize
x_test = x_test.astype('float32') / 255.0

# Predict on first 5 test images
predictions = model.predict(x_test[:5])

# Class names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# Show results
for i in range(5):
    predicted_class = classes[np.argmax(predictions[i])]
    actual_class = classes[y_test[i][0]]
    confidence = np.max(predictions[i]) * 100
    print(f"Image {i+1}: Predicted={predicted_class} ({confidence:.1f}%), Actual={actual_class}")
```

Save this as `test_prediction.py` and run:
```powershell
.\venv\Scripts\python.exe test_prediction.py
```

## What the Model Can Recognize

The model is trained on CIFAR-10, which includes these 10 classes:
1. ✈️ Airplane
2. 🚗 Automobile
3. 🐦 Bird
4. 🐱 Cat
5. 🦌 Deer
6. 🐕 Dog
7. 🐸 Frog
8. 🐴 Horse
9. 🚢 Ship
10. 🚚 Truck

## Expected Accuracy

For a quick 5-epoch training:
- Training Accuracy: ~60-70%
- Validation Accuracy: ~55-65%

This is normal for a quick training run. For better accuracy, train for 20-50 epochs.

## Troubleshooting

### "Model file not found"
- Make sure training completed successfully
- Check that `models/custom_cnn.h5` exists
- Run: `ls models/` to see available models

### "Image not found"
- Check the image path is correct
- Use absolute path: `C:\full\path\to\image.jpg`
- Supported formats: .jpg, .jpeg, .png, .bmp

### Low accuracy predictions
- The model was trained for only 5 epochs (quick demo)
- For better results, train longer:
  ```powershell
  .\venv\Scripts\python.exe scripts\train_custom_cnn.py --dataset cifar10 --epochs 30
  ```

## Next Steps After First Prediction

1. **Try more images** - Test with different CIFAR-10 classes
2. **Train longer** - Use 20-50 epochs for better accuracy
3. **Try transfer learning** - Use pre-trained models like ResNet50:
   ```powershell
   .\venv\Scripts\python.exe scripts\train_transfer_learning.py --model resnet50 --dataset cifar10 --epochs 15
   ```
4. **Use your own dataset** - Organize images in folders and train on custom data

## Quick Commands Reference

```powershell
# Navigate to project
cd C:\Users\BhaktiVaidya\Box\Bob_Agent\dl_image_classification

# Predict with visualization
.\venv\Scripts\python.exe scripts\predict.py --image path/to/image.jpg --model models/custom_cnn.h5 --visualize

# Predict on multiple images
.\venv\Scripts\python.exe scripts\predict.py --image path/to/folder --model models/custom_cnn.h5 --batch

# View model summary
.\venv\Scripts\python.exe -c "from tensorflow import keras; model = keras.models.load_model('models/custom_cnn.h5'); model.summary()"

# Check available models
ls models/
```

---
**Remember:** The training must complete before you can run predictions!
Monitor Terminal 1 for training progress.