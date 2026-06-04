@echo off
REM Batch script to run Python scripts with the virtual environment

REM Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run: py -3.11 -m venv venv
    pause
    exit /b 1
)

REM Run the Python script with virtual environment
venv\Scripts\python.exe %*

REM Example usage:
REM   run_with_venv.bat scripts\train_custom_cnn.py --dataset cifar10 --epochs 5
REM   run_with_venv.bat scripts\predict.py --model models\custom_cnn.h5 --dataset cifar10

@REM Made with Bob
