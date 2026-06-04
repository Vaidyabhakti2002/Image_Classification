# Python Setup Guide for TensorFlow Project

## Problem
TensorFlow does not support Python 3.14. You currently have Python 3.14.4 installed, which is why you're getting the `ModuleNotFoundError: No module named 'tensorflow'` error.

## Solution: Install Python 3.11

### Step 1: Download Python 3.11
1. Go to https://www.python.org/downloads/
2. Download Python 3.11.x (latest 3.11 version)
3. During installation, **check "Add Python 3.11 to PATH"**

### Step 2: Verify Installation
After installing Python 3.11, open a new PowerShell window and run:
```powershell
py --list
```
You should see both Python 3.14 and 3.11 listed.

### Step 3: Create Virtual Environment with Python 3.11
Navigate to the project directory and create a virtual environment:
```powershell
cd C:\Users\BhaktiVaidya\Box\Bob_Agent\dl_image_classification
py -3.11 -m venv venv
```

### Step 4: Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then try activating again.

### Step 5: Install Requirements
With the virtual environment activated:
```powershell
pip install -r requirements.txt
```

### Step 6: Run the Prediction Script
```powershell
python scripts\predict.py
```

## Alternative: Use Conda (Recommended for Data Science)

If you have Anaconda or Miniconda installed:

```powershell
# Create environment with Python 3.11
conda create -n tf_env python=3.11

# Activate environment
conda activate tf_env

# Navigate to project
cd C:\Users\BhaktiVaidya\Box\Bob_Agent\dl_image_classification

# Install requirements
pip install -r requirements.txt

# Run script
python scripts\predict.py
```

## Quick Reference: Python Version Compatibility

| TensorFlow Version | Python Support |
|-------------------|----------------|
| 2.13.x - 2.15.x   | 3.8 - 3.11    |
| 2.16.x            | 3.9 - 3.12    |
| Future versions   | May support 3.13+ |

**Note:** Python 3.14 is very new (released 2026) and most ML libraries haven't added support yet.

## Troubleshooting

### Issue: "py -3.11" not found
- Python 3.11 is not installed. Follow Step 1.

### Issue: Virtual environment won't activate
- Check execution policy (see Step 4)
- Try running PowerShell as Administrator

### Issue: Still getting TensorFlow errors after installation
- Make sure virtual environment is activated (you should see `(venv)` in your prompt)
- Verify Python version: `python --version` (should show 3.11.x)
- Reinstall TensorFlow: `pip install --upgrade tensorflow`

## Next Steps After Setup

Once TensorFlow is installed successfully:

1. **Train a model first** (you need a trained model before prediction):
   ```powershell
   python scripts\train_custom_cnn.py
   # or
   python scripts\train_transfer_learning.py
   ```

2. **Then run predictions**:
   ```powershell
   python scripts\predict.py
   ```

3. **Check the QUICKSTART.md** for detailed usage instructions