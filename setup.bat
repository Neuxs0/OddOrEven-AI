@echo off
prompt $
cls
title OddOrEven Setup [ Checking Python version ]
echo Checking Python Version...
for /f "tokens=2 delims==" %%i in ('python -c "import sys; print(sys.version_info >= (3, 10))"') do set python_version=%%i

if "%python_version%" == "False" (
    cls
    title OddOrEven Setup [ Error ]
    echo Error: Install Python version 3.10 or higher.
    pause
    exit /b
)

cls
title OddOrEven Setup [ Installing Required Libraries ]
echo Installing required libraries: numpy, CuPy, nvidia-ml-py3, psutil
pause
pip install numpy
cls
pip install CuPy
cls
pip install nvidia-ml-py3
cls
pip install psutil
cls

title OddOrEven Setup [ Done ]
echo Setup successful, now you can run train.bat or run.bat
pause