# Complete Usage Guide - Deep Learning Image Classification

## ✅ Setup Complete!

Your environment is ready:
- ✅ Python 3.11.9 installed
- ✅ Virtual environment created at `venv\`
- ✅ TensorFlow and all dependencies installed
- ✅ Training is currently running in Terminal 1

---

## 🚀 How to Run Scripts (3 Methods)

### Method 1: Full Path (Most Reliable) ⭐ RECOMMENDED

```powershell
# Navigate to project
cd C:\Users\BhaktiVaidya\Box\Bob_Agent\dl_image_classification

# Run training
.\venv\Scripts\python.exe scripts\train_custom_cnn.py --dataset cifar10 --epochs 5

# Run predictions
.\venv\Scripts\python.exe scripts\predict.py --model models\custom_cnn.h5 --dataset cifar10 --visualize
```

### Method 2: Batch Script (Easiest)

```powershell
cd C:\Users\BhaktiVaidya\Box\Bob_Agent\dl_image_classification

# Run training
.\run_with_venv.bat scripts\train_custom_cnn.py --dataset cifar10 --epochs 5

# Run predictions
.\run_with_venv.bat scripts\predict.py --model models\custom_cnn.h5 --dataset cifar10
```

### Method 3: Activate Virtual Environment

```powershell
cd C:\Users\BhaktiVaidya\Box\Bob_Agent\dl_image_classification

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Now you can use 'python' directly (venv is active)
python scripts\train_custom_cnn.py --dataset cifar10 --epochs 5
python scripts\predict.py --model models\custom_cnn.h5 --dataset cifar10

# Deactivate when done
deactivate
```

---

## 📋 Step-by-Step: Your First Prediction

### Step 1: Wait for Training to Complete

Monitor Terminal 1. You'll see:
```
Epoch 1/5
Epoch 2/5
Epoch 3/5
Epoch 4/5
Epoch 5/5
Model saved to: models/custom_cnn.h5
```

**Estimated time:** 5-10 minutes

### Step 2: Verify Model Exists

```powershell
cd C:\Users\BhaktiVaidya\Box\Bob_Agent\dl_image_classification
ls models\
```

You should see `custom_cnn.h5`

### Step 3: Run Your First Prediction

**Option A: Test on CIFAR-10 images**
```powershell
.\venv\Scripts\python.exe scripts\predict.py --model models\custom_cnn.h5 --dataset cifar10
```

**Option B: Test on your own image**
```powershell
.\venv\Scripts\python.exe scripts\predict.py --image "C:\path\to\your\image.jpg" --model models\custom_cnn.h5 --visualize
```

---

## 🎯 Common Commands

### Training Commands

```powershell
# Quick training (5 epochs)
.\venv\Scripts\python.exe scripts\train_custom_cnn.py --dataset cifar10 --epochs 5

# Better accuracy (30 epochs)
.\venv\Scripts\python.exe scripts\train_custom_cnn.py --dataset cifar10 --epochs 30

# Transfer learning with ResNet50
.\venv\Scripts\python.exe scripts\train_transfer_learning.py --model resnet50 --dataset cifar10 --epochs 15

# Fashion-MNIST dataset
.\venv\Scripts\python.exe scripts\train_custom_cnn.py --dataset fashion_mnist --epochs 20
```

### Prediction Commands

```powershell
# Predict single image with visualization
.\venv\Scripts\python.exe scripts\predict.py --image image.jpg --model models\custom_cnn.h5 --visualize

# Predict multiple images in a folder
.\venv\Scripts\python.exe scripts\predict.py --image C:\path\to\folder --model models\custom_cnn.h5 --batch

# Show top 10 predictions
.\venv\Scripts\python.exe scripts\predict.py --image image.jpg --model models\custom_cnn.h5 --top_k 10
```

### Utility Commands

```powershell
# Check Python version (should be 3.11.9)
.\venv\Scripts\python.exe --version

# List installed packages
.\venv\Scripts\pip.exe list

# View model summary
.\venv\Scripts\python.exe -c "from tensorflow import keras; model = keras.models.load_model('models/custom_cnn.h5'); model.summary()"

# Start TensorBoard
.\venv\Scripts\tensorboard.exe --logdir=logs
```

---

## 🎨 What the Model Can Recognize

The CIFAR-10 model recognizes these 10 classes:

1. ✈️ **Airplane** - Commercial aircraft, jets
2. 🚗 **Automobile** - Cars, sedans, vehicles
3. 🐦 **Bird** - Various bird species
4. 🐱 **Cat** - Domestic cats
5. 🦌 **Deer** - Deer and similar animals
6. 🐕 **Dog** - Domestic dogs
7. 🐸 **Frog** - Frogs and amphibians
8. 🐴 **Horse** - Horses and ponies
9. 🚢 **Ship** - Boats, ships, vessels
10. 🚚 **Truck** - Trucks, lorries

---

## ⚠️ Common Mistakes to Avoid

### ❌ WRONG - Using System Python
```powershell
python scripts\train_custom_cnn.py  # Uses Python 3.14 - NO TensorFlow!
```

### ✅ CORRECT - Using Virtual Environment Python
```powershell
.\venv\Scripts\python.exe scripts\train_custom_cnn.py  # Uses Python 3.11 - Has TensorFlow!
```

---

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'tensorflow'"

**Cause:** You're using system Python 3.14 instead of venv Python 3.11

**Solution:** Use the full path:
```powershell
.\venv\Scripts\python.exe your_script.py
```

### Error: "Model file not found"

**Cause:** Training hasn't completed yet

**Solution:** Wait for training to finish and save the model

### Error: "Cannot activate virtual environment"

**Cause:** PowerShell execution policy

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Low Prediction Accuracy

**Cause:** Model trained for only 5 epochs (quick demo)

**Solution:** Train for more epochs:
```powershell
.\venv\Scripts\python.exe scripts\train_custom_cnn.py --dataset cifar10 --epochs 30
```

---

## 📊 Expected Performance

### Quick Training (5 epochs)
- Training Time: 5-10 minutes
- Training Accuracy: ~60-70%
- Validation Accuracy: ~55-65%

### Full Training (30 epochs)
- Training Time: 30-60 minutes
- Training Accuracy: ~85-90%
- Validation Accuracy: ~75-80%

### Transfer Learning (ResNet50, 15 epochs)
- Training Time: 20-30 minutes
- Training Accuracy: ~90-95%
- Validation Accuracy: ~85-90%

---

## 🎓 Next Steps

1. ✅ **Wait for current training to complete** (Terminal 1)
2. ✅ **Run your first prediction** using the commands above
3. ⚡ **Experiment with different models** (ResNet50, VGG16)
4. ⚡ **Train on Fashion-MNIST** for clothing classification
5. ⚡ **Use your own images** for predictions
6. ⚡ **Train longer** (30-50 epochs) for better accuracy
7. ⚡ **Try data augmentation** for improved performance

---

## 📁 Project Structure

```
dl_image_classification/
├── venv/                          # Virtual environment (Python 3.11)
├── models/                        # Saved models (.h5 files)
│   └── custom_cnn.h5             # Your trained model
├── logs/                          # TensorBoard logs
├── scripts/
│   ├── train_custom_cnn.py       # Train custom CNN
│   ├── train_transfer_learning.py # Transfer learning
│   └── predict.py                # Make predictions
├── utils/                         # Helper functions
├── config.py                      # Configuration
├── requirements.txt               # Dependencies
├── run_with_venv.bat             # Batch script helper
└── COMPLETE_USAGE_GUIDE.md       # This file
```

---

## 💡 Pro Tips

1. **Always use the virtual environment** - Never use system Python
2. **Monitor training** - Watch Terminal 1 for progress
3. **Save checkpoints** - Best models are automatically saved
4. **Use TensorBoard** - Visualize training progress
5. **Start small** - Test with 5 epochs, then increase
6. **GPU not required** - CPU training works fine for CIFAR-10
7. **Backup models** - Copy `.h5` files to safe location

---

## 🆘 Quick Reference Card

```powershell
# Navigate to project
cd C:\Users\BhaktiVaidya\Box\Bob_Agent\dl_image_classification

# Train model (5 epochs)
.\venv\Scripts\python.exe scripts\train_custom_cnn.py --dataset cifar10 --epochs 5

# Predict on image
.\venv\Scripts\python.exe scripts\predict.py --image image.jpg --model models\custom_cnn.h5 --visualize

# Check Python version
.\venv\Scripts\python.exe --version

# List models
ls models\

# View TensorBoard
.\venv\Scripts\tensorboard.exe --logdir=logs
```

---

**Remember:** The training in Terminal 1 is working perfectly! Just wait for it to complete, then run predictions using the commands above. 🎉