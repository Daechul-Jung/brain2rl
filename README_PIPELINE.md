# Integrated Classification and Tokenization Pipeline

This pipeline provides an integrated solution for classifying time series sensor data and generating tokens suitable for reinforcement learning trajectories.

## Overview

The pipeline consists of two main components:

1. **Action Classifier**: A CNN-based model that classifies sensor data into different action categories
2. **Brain Tokenizer**: A transformer-based model that converts classified data into tokens with Q/K/V matrices

## Features

- **Flexible Data Loading**: Automatically detects and loads CSV files with sensor data
- **Automatic Preprocessing**: Handles data normalization and label encoding
- **Sliding Window Processing**: Processes time series data with configurable window sizes and overlap
- **Model Persistence**: Saves trained models and generated tokens
- **Training Visualization**: Generates plots of training progress
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Data Format

Your sensor data should be in CSV format with the following structure:

```csv
acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,label
0.1,0.2,-0.1,0.05,0.03,-0.02,0
0.15,0.18,-0.08,0.06,0.04,-0.01,0
...
```

**Required columns:**
- Sensor data columns (e.g., `acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`, `gyro_z`)
- `label` column with action categories (0, 1, 2, etc.)

**File naming convention:**
- Files should be named as `SUBJ_001.csv`, `SUBJ_002.csv`, etc.
- Or any descriptive name ending with `.csv`

## Installation

1. Ensure you have the required dependencies:
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib tqdm
```

2. Clone or download the pipeline files to your project directory.

## Usage

### Quick Start

Run the pipeline with default settings:

```bash
python run_pipeline.py --data-dir /path/to/your/sensor/data
```

### Advanced Usage

Run with custom configuration:

```bash
python run_pipeline.py \
    --data-dir /path/to/your/sensor/data \
    --config config/pipeline_config.json \
    --output-dir my_results \
    --log-level DEBUG
```

Process specific subjects:

```bash
python run_pipeline.py \
    --data-dir /path/to/your/sensor/data \
    --subject-ids SUBJ_001 SUBJ_002 SUBJ_003
```

### Configuration

The pipeline uses a JSON configuration file (`config/pipeline_config.json`) with the following parameters:

```json
{
    "window_size": 100,
    "batch_size": 32,
    "classifier_lr": 0.001,
    "classifier_epochs": 100,
    "classifier_dropout": 0.3,
    "tokenizer_lr": 0.0001,
    "tokenizer_epochs": 50,
    "tokenizer_dropout": 0.1,
    "n_tokens": 512,
    "embedding_dim": 128,
    "nhead": 8,
    "num_encoder_layers": 6
}
```

**Key Parameters:**
- `window_size`: Size of sliding window for time series processing
- `batch_size`: Training batch size
- `classifier_epochs`: Number of training epochs for the classifier
- `tokenizer_epochs`: Number of training epochs for the tokenizer
- `n_tokens`: Number of tokens in the vocabulary
- `embedding_dim`: Dimension of the embedding space

## Pipeline Steps

1. **Data Loading**: Loads CSV files from the specified directory
2. **Preprocessing**: Normalizes sensor data and encodes action labels
3. **Data Splitting**: Creates train/validation/test splits
4. **Classifier Training**: Trains the CNN action classifier
5. **Tokenizer Training**: Trains the transformer-based tokenizer
6. **Token Generation**: Generates tokens from the test dataset
7. **Results Storage**: Saves models, tokens, and training plots

## Output Files

After successful execution, the pipeline creates:

```
output/
├── pipeline_results.pth          # Complete pipeline results
├── generated_tokens.npz          # Generated tokens for RL
├── pipeline_config.json          # Configuration used
├── classifier_training.png       # Classifier training plots
└── tokenizer_training.png        # Tokenizer training plots

models/
├── classification/
│   └── best_classifier.pth       # Best trained classifier
└── tokenization/
    └── best_tokenizer.pth        # Best trained tokenizer
```

## Testing

Test the pipeline with sample data:

```bash
python test_integrated_pipeline.py
```

This will:
1. Create sample sensor data
2. Test individual pipeline components
3. Run the complete pipeline
4. Verify token generation

## Model Architecture

### Action Classifier

The classifier uses a 1D CNN architecture:
- **Temporal Convolution**: Extracts temporal features from sensor data
- **Feature Extraction**: Multiple convolutional layers with increasing channel counts
- **Classification Head**: Fully connected layers for final action prediction

### Brain Tokenizer

The tokenizer combines:
- **CNN Feature Extractor**: Converts sensor data to embeddings
- **Transformer Encoder**: Models temporal dependencies
- **Token Projection**: Maps embeddings to token space

## Integration with Reinforcement Learning

The generated tokens can be used in RL algorithms:

```python
import numpy as np

# Load generated tokens
token_data = np.load('output/generated_tokens.npz')
tokens = token_data['tokens']
labels = token_data['labels']

# Use tokens in your RL algorithm
# tokens shape: (n_sequences, sequence_length, n_tokens)
# Each token sequence represents a trajectory of actions
```

## Troubleshooting

### Common Issues

1. **No CSV files found**: Ensure your data directory contains `.csv` files
2. **Memory errors**: Reduce `batch_size` or `window_size` in configuration
3. **Training not converging**: Adjust learning rates or increase epochs
4. **CUDA out of memory**: Use smaller models or reduce batch size

### Debug Mode

Run with debug logging for detailed information:

```bash
python run_pipeline.py --data-dir /path/to/data --log-level DEBUG
```

### Check Logs

Review the `pipeline.log` file for detailed execution information and error messages.

## Performance Tips

1. **Data Size**: Larger datasets generally lead to better model performance
2. **Window Size**: Choose window size based on your action duration
3. **Batch Size**: Use the largest batch size that fits in memory
4. **Epochs**: Monitor validation metrics to avoid overfitting

## Customization

### Adding New Sensor Types

Modify the `load_sensor_data` method in `IntegratedPipeline` to handle different sensor column names.

### Custom Model Architectures

Extend the `ActionClassifier` and `BrainTokenizer` classes to implement your own architectures.

### Custom Loss Functions

Modify the training loops to use different loss functions or training objectives.

## Support

For issues or questions:
1. Check the log files for error details
2. Verify your data format matches the expected structure
3. Test with the provided sample data first
4. Review the configuration parameters

## Next Steps

After successful token generation:
1. Integrate tokens with your RL algorithm
2. Fine-tune model parameters based on your specific data
3. Extend the pipeline for real-time processing
4. Add additional preprocessing steps as needed
