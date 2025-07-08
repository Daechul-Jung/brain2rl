@echo off
REM Brain2RL Pipeline Batch Script for Windows
REM ==========================================
REM 
REM This script provides easy commands for running the Brain2RL pipeline
REM on Windows systems. Choose from the options below.

echo.
echo ===============================================
echo           Brain2RL Pipeline Runner
echo ===============================================
echo.
echo Choose an option:
echo 1. Generate synthetic data and run full pipeline
echo 2. Run individual pipeline components
echo 3. Quick test with mock simulation
echo 4. Full pipeline with real data
echo 5. Development mode (reduced parameters)
echo 6. Exit
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto full_synthetic
if "%choice%"=="2" goto individual
if "%choice%"=="3" goto quick_test
if "%choice%"=="4" goto full_real
if "%choice%"=="5" goto dev_mode
if "%choice%"=="6" goto exit

:full_synthetic
echo.
echo Running full pipeline with synthetic data...
echo.

REM Create directories
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "results" mkdir results

REM Generate synthetic data
echo Step 1/2: Generating synthetic sensor data...
python brain2rl/cli.py generate-data --n-samples 1000 --n-channels 32 --n-timesteps 512 --n-classes 6 --output-path data/synthetic_sensor_data.npz

REM Run full pipeline
echo Step 2/2: Running full Brain2RL pipeline...
python brain2rl/cli.py full --data-path data/synthetic_sensor_data.npz --output-dir results/ --device auto

echo.
echo Pipeline completed! Check results/ directory for outputs.
goto end

:individual
echo.
echo Running individual pipeline components...
echo.

REM Check if synthetic data exists
if not exist "data/synthetic_sensor_data.npz" (
    echo Generating synthetic data first...
    python brain2rl/cli.py generate-data --n-samples 500 --output-path data/synthetic_sensor_data.npz
)

REM Create directories
if not exist "models" mkdir models
if not exist "results" mkdir results

REM Step 1: Classification
echo Step 1/4: Training classification model...
python brain2rl/cli.py classification --mode train --data-path data/synthetic_sensor_data.npz --model-path models/classifier.pth --epochs 50

echo Step 1b/4: Classifying data...
python brain2rl/cli.py classification --mode classify --data-path data/synthetic_sensor_data.npz --model-path models/classifier.pth --output-path results/classified_data.npz

REM Step 2: Tokenization
echo Step 2/4: Training tokenizer...
python brain2rl/cli.py tokenization --mode train --classified-data results/classified_data.npz --model-path models/tokenizer.pth --epochs 50

echo Step 2b/4: Generating tokens...
python brain2rl/cli.py tokenization --mode tokenize --classified-data results/classified_data.npz --model-path models/tokenizer.pth --output-path results/tokens.npz

REM Step 3: RL Training
echo Step 3/4: Training RL agent...
python brain2rl/cli.py rl-training --token-data results/tokens.npz --episodes 500 --model-path models/rl_agent.pth --plot-results

REM Step 4: Simulation
echo Step 4/4: Running simulation...
python brain2rl/cli.py simulation --model-path models/rl_agent.pth --token-data results/tokens.npz --episodes 5 --visualize --save-data results/simulation_data.npz

echo.
echo Individual components completed! Check models/ and results/ directories.
goto end

:quick_test
echo.
echo Running quick test with mock simulation...
echo.

REM Create directories
if not exist "data" mkdir data
if not exist "results" mkdir results

REM Generate small test data
echo Generating test data...
python brain2rl/cli.py generate-data --n-samples 100 --n-channels 16 --n-timesteps 256 --output-path data/test_data.npz

REM Run quick pipeline
echo Running quick pipeline test...
python brain2rl/cli.py full --data-path data/test_data.npz --output-dir results/test/ --device cpu

REM Mock simulation
echo Running mock simulation...
python brain2rl/cli.py simulation --model-path results/test/full_pipeline_results.pth --episodes 3 --mock-mode --use-gazebo false --use-ros false

echo.
echo Quick test completed!
goto end

:full_real
echo.
echo Running full pipeline with real data...
echo.

set /p data_path="Enter path to your sensor data file: "

if not exist "%data_path%" (
    echo Error: Data file not found!
    goto end
)

REM Create directories
if not exist "models" mkdir models
if not exist "results" mkdir results

REM Run full pipeline
echo Running full Brain2RL pipeline with real data...
python brain2rl/cli.py full --data-path "%data_path%" --output-dir results/real_data/ --device auto

echo.
echo Real data pipeline completed! Check results/real_data/ directory.
goto end

:dev_mode
echo.
echo Running development mode with reduced parameters...
echo.

REM Create directories
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "results" mkdir results

REM Generate dev data
echo Generating development data...
python brain2rl/cli.py generate-data --n-samples 200 --n-channels 8 --n-timesteps 128 --output-path data/dev_data.npz

REM Quick classification
echo Training classification (reduced)...
python brain2rl/cli.py classification --mode train --data-path data/dev_data.npz --model-path models/dev_classifier.pth --epochs 10 --batch-size 16

python brain2rl/cli.py classification --mode classify --data-path data/dev_data.npz --model-path models/dev_classifier.pth --output-path results/dev_classified.npz

REM Quick tokenization
echo Training tokenization (reduced)...
python brain2rl/cli.py tokenization --mode train --classified-data results/dev_classified.npz --model-path models/dev_tokenizer.pth --epochs 10 --batch-size 16

python brain2rl/cli.py tokenization --mode tokenize --classified-data results/dev_classified.npz --model-path models/dev_tokenizer.pth --output-path results/dev_tokens.npz

REM Quick RL training
echo Training RL agent (reduced)...
python brain2rl/cli.py rl-training --token-data results/dev_tokens.npz --episodes 50 --model-path models/dev_rl_agent.pth --batch-size 32

REM Mock simulation
echo Running development simulation...
python brain2rl/cli.py simulation --model-path models/dev_rl_agent.pth --token-data results/dev_tokens.npz --episodes 2 --mock-mode

echo.
echo Development mode completed!
goto end

:exit
echo Exiting...
goto end

:end
echo.
echo ===============================================
echo        Brain2RL Pipeline Script Complete
echo ===============================================
echo.

REM Show available results
if exist "results" (
    echo Available results:
    dir /b results
    echo.
)

if exist "models" (
    echo Available models:
    dir /b models
    echo.
)

echo Thank you for using Brain2RL!
echo For more information, see brain2rl/README.md
echo.
pause 