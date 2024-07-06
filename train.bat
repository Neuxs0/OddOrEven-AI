@echo off
prompt $
cls
title OddOrEven AI Training [ Getting Ready ]
set values=""

goto :warningGOTO

:checkInt
setlocal
set "input=%~1"
set /a num=%input% 2>nul
if "%input%" neq "%num%" (
    endlocal & set "error=1"
) else (
    endlocal & set "error=0"
)
goto :eof

:checkFloat
echo %~1 | python -c "import sys; sys.exit(1 if not isinstance(float(sys.stdin.read().strip()), float) else 0)"
if errorlevel 1 (
    set "error=1"
) else (
    set "error=0"
)
goto :eof

:warningGOTO
cls
echo WARNING: Beware of overfitting and underfitting.
echo Do /back when you want to go back a question
echo.
echo Overfitting is the AI memorizing the data rather than learning
echo Underfitting is the AI not being able to learn the data due to too little neurons or layers
echo.
pause
cls

:hiddenLayerCountGOTO
echo How many Hidden Layers do you want? (Default is 2)
set /p hiddenLayerCount=
if /i "%hiddenLayerCount%" == "/back" (
    cls
    goto warningGOTO
)
call :checkInt %hiddenLayerCount%
if "%error%" == "1" (
    cls
    echo Invalid input. Please enter an integer value.
    echo.
    goto hiddenLayerCountGOTO
)
if %hiddenLayerCount% leq 0 (
    cls
    echo Invalid input. Hidden layer count must be greater than 0.
    echo.
    goto hiddenLayerCountGOTO
)
cls

:hiddenLayerNeuronCountGOTO
echo Input as a list: 64, 48, 32, 24, etc...
echo.
echo How many Hidden Layer Neurons do you want? (Default is 24, 16)
set /p hiddenLayerNeuronCount=
if /i "%hiddenLayerNeuronCount%" == "/back" (
    cls
    goto hiddenLayerCountGOTO
)

:: Remove all spaces from hiddenLayerNeuronCount
set "hiddenLayerNeuronCount=%hiddenLayerNeuronCount: =%"

set "validList=1"
set "neuronCount=0"
for %%a in (%hiddenLayerNeuronCount%) do (
    call :checkInt %%a
    if "%error%" == "1" (
        set "validList=0"
        cls
        echo Invalid input. Each hidden layer neuron count must be an integer.
        echo.
        goto hiddenLayerNeuronCountGOTO
    )
    set /a neuronCount+=1
)

if %neuronCount% neq %hiddenLayerCount% (
    cls
    echo Invalid input. Number of neuron counts must match the number of hidden layers.
    echo.
    goto hiddenLayerNeuronCountGOTO
)
cls

:learningRateGOTO
echo What learning rate? (Default is 0.01)
set /p learningRate=
if /i "%learningRate%" == "/back" (
    cls
    goto hiddenLayerNeuronCountGOTO
)
call :checkFloat %learningRate%
if "%error%" == "1" (
    cls
    echo Invalid input. Please enter a float value.
    echo.
    goto learningRateGOTO
)
cls

:epochsGOTO
echo How many epochs do you want? (Default is 100)
set /p epochs=
if /i "%epochs%" == "/back" (
    cls
    goto learningRateGOTO
)
call :checkInt %epochs%
if "%error%" == "1" (
    cls
    echo Invalid input. Please enter an integer value.
    echo.
    goto epochsGOTO
)
cls

:batchSizeGOTO
echo What is the batch size? (Default is 10)
set /p batchSize=
if /i "%batchSize%" == "/back" (
    cls
    goto epochsGOTO
)
call :checkInt %batchSize%
if "%error%" == "1" (
    cls
    echo Invalid input. Please enter an integer value.
    echo.
    goto batchSizeGOTO
)
cls

set inputLayerNeuronCount=32
set outputLayerNeuronCount=2

title OddOrEven AI Training [ Training... ]
py .\\src\\train.py %inputLayerNeuronCount% %outputLayerNeuronCount% %hiddenLayerCount% [%hiddenLayerNeuronCount%] %learningRate% %epochs% %batchSize%
pause
