@echo off
echo ========================================
echo  Counterfactual MRI - Setup
echo ========================================
echo.

echo [1/5] Checking Python 3.11...
py -3.11 --version
if errorlevel 1 (
    echo ERROR: Python 3.11 not found.
    echo Please install from: https://www.python.org/downloads/release/python-3119/
    pause
    exit /b 1
)

echo.
echo [2/5] Installing PyTorch (CUDA 12.1)... (~2GB, takes 10min)
py -3.11 -m pip install torch --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo Retrying with cu124...
    py -3.11 -m pip install torch --index-url https://download.pytorch.org/whl/cu124
)

echo.
echo [3/5] Installing diffusers / transformers...
py -3.11 -m pip install diffusers transformers accelerate

echo.
echo [4/5] Installing xformers...
py -3.11 -m pip install xformers --index-url https://download.pytorch.org/whl/cu121

echo.
echo [5/5] Installing other libraries...
py -3.11 -m pip install numpy scipy scikit-image Pillow matplotlib

echo.
echo ========================================
echo  Setup complete!
echo  Run with:
echo    py -3.11 demo.py --use_synthetic
echo    py -3.11 demo.py --use_synthetic --use_real_sd
echo ========================================
pause
