# Integrated Classification and Tokenization Pipeline - Complete Summary

## 🎯 What Has Been Created

I've successfully created a complete integrated pipeline that combines action classification and tokenization for your time series sensor data. This pipeline is designed to process your machine-generated sensor data and generate tokens suitable for reinforcement learning trajectories.

## 📁 File Structure

```
brain2rl/
├── core/
│   └── integrated_classification_tokenization.py  # Main pipeline implementation
├── config/
│   └── pipeline_config.json                       # Configuration file
├── test_integrated_pipeline.py                    # Test script with sample data
├── run_pipeline.py                                # Command-line interface
├── run_on_your_data.py                           # Simple script for your data
├── example_rl_integration.py                     # RL integration example
├── README_PIPELINE.md                            # Detailed documentation
└── PIPELINE_SUMMARY.md                           # This summary
```

## 🚀 Key Features

### 1. **Integrated Pipeline**
- **Action Classifier**: CNN-based model for classifying sensor data into actions
- **Brain Tokenizer**: Transformer-based model for generating tokens
- **Automatic Data Processing**: Handles CSV files with sensor data
- **Sliding Window Processing**: Configurable window sizes for time series data

### 2. **Flexible Data Loading**
- Automatically detects CSV files in your data directory
- Supports various sensor column names (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
- Handles multiple subjects/experiments
- Automatic data preprocessing and normalization

### 3. **Model Architecture**
- **Classifier**: 1D CNN with temporal convolutions and feature extraction
- **Tokenizer**: CNN feature extractor + Transformer encoder + Token projection
- **Configurable**: Adjustable hyperparameters for different data types

### 4. **Output Generation**
- Trained models saved to `models/` directory
- Generated tokens saved as `.npz` files
- Training plots and metrics
- Complete pipeline results

## 🔧 How to Use

### Quick Start with Your Data

1. **Update the data path** in `run_on_your_data.py`:
   ```python
   data_dir = "path/to/your/sensor/data"  # Change this line
   ```

2. **Run the pipeline**:
   ```bash
   python3 run_on_your_data.py
   ```

### Command-Line Interface

For more control, use the command-line interface:

```bash
# Run with default settings
python3 run_pipeline.py --data-dir /path/to/your/data

# Run with custom configuration
python3 run_pipeline.py --data-dir /path/to/your/data --config config/pipeline_config.json

# Process specific subjects
python3 run_pipeline.py --data-dir /path/to/your/data --subject-ids SUBJ_001 SUBJ_002
```

### Data Format Requirements

Your sensor data should be in CSV format:

```csv
acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,label
0.1,0.2,-0.1,0.05,0.03,-0.02,0
0.15,0.18,-0.08,0.06,0.04,-0.01,0
...
```

**Required columns:**
- Sensor data columns (adjust names as needed)
- `label` column with action categories (0, 1, 2, etc.)

## 🧪 Testing

Test the pipeline with sample data:

```bash
python3 test_integrated_pipeline.py
```

This will:
1. Create sample sensor data
2. Test individual components
3. Run the complete pipeline
4. Verify token generation

## 🔄 Pipeline Workflow

1. **Data Loading**: Loads CSV files from your data directory
2. **Preprocessing**: Normalizes sensor data and encodes action labels
3. **Data Splitting**: Creates train/validation/test splits
4. **Classifier Training**: Trains the CNN action classifier
5. **Tokenizer Training**: Trains the transformer-based tokenizer
6. **Token Generation**: Generates tokens from the test dataset
7. **Results Storage**: Saves models, tokens, and training plots

## 🎮 Reinforcement Learning Integration

After running the pipeline, you can integrate the generated tokens with RL algorithms:

```bash
python3 example_rl_integration.py
```

This script demonstrates:
- Loading generated tokens
- Creating RL state representations
- Building trajectory datasets
- Integration examples for PPO and SAC algorithms

## 📊 Output Files

After successful execution:

```
output/
├── pipeline_results.pth          # Complete pipeline results
├── generated_tokens.npz          # Generated tokens for RL
├── classifier_training.png       # Classifier training plots
└── tokenizer_training.png        # Tokenizer training plots

models/
├── classification/
│   └── best_classifier.pth       # Best trained classifier
└── tokenization/
    └── best_tokenizer.pth        # Best trained tokenizer
```

## ⚙️ Configuration

Key parameters you can adjust in `config/pipeline_config.json`:

- `window_size`: Size of sliding window for time series processing
- `batch_size`: Training batch size
- `classifier_epochs`: Number of training epochs for classifier
- `tokenizer_epochs`: Number of training epochs for tokenizer
- `n_tokens`: Number of tokens in vocabulary
- `embedding_dim`: Dimension of the embedding space

## 🔍 Troubleshooting

### Common Issues

1. **No CSV files found**: Ensure your data directory contains `.csv` files
2. **Memory errors**: Reduce `batch_size` or `window_size`
3. **Training not converging**: Adjust learning rates or increase epochs
4. **CUDA out of memory**: Use smaller models or reduce batch size

### Debug Mode

Run with debug logging:

```bash
python3 run_pipeline.py --data-dir /path/to/data --log-level DEBUG
```

### Check Logs

Review the `pipeline.log` file for detailed execution information.

## 🎯 Next Steps

1. **Run on your data**: Update the data path and run `run_on_your_data.py`
2. **Test integration**: Run `example_rl_integration.py` to see RL integration
3. **Customize**: Adjust configuration parameters for your specific data
4. **Extend**: Modify the pipeline for additional preprocessing or model architectures
5. **RL Training**: Use the generated tokens to train your reinforcement learning agent

## 📚 Additional Resources

- **README_PIPELINE.md**: Comprehensive documentation
- **Test scripts**: Verify functionality with sample data
- **Configuration files**: Adjust parameters for your needs
- **Example integrations**: See how to use tokens with RL algorithms

## 🎉 Summary

You now have a complete, working pipeline that:

✅ **Loads and preprocesses** your sensor data  
✅ **Trains an action classifier** using CNN architecture  
✅ **Generates tokens** using transformer-based tokenization  
✅ **Stores results** for reinforcement learning integration  
✅ **Provides examples** for RL algorithm integration  

The pipeline is ready to process your time series sensor data and generate tokens that can be used in reinforcement learning trajectories. Simply update the data path and run the pipeline to get started!
