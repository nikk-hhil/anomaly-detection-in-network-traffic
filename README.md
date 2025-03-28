# Network Traffic Anomaly Detection

This project implements a machine learning system to detect anomalies (malicious activity) in network traffic data using the CIC-IDS 2017 dataset.

## Project Structure

```
network_anomaly_detection/
│
├── data/                      # Directory for dataset files
│   └── processed/             # Processed and merged datasets
│
├── src/
│   ├── __init__.py            # Makes the directory a Python package
│   ├── data_loader.py         # For loading and merging data
│   ├── preprocessor.py        # Data cleaning and preprocessing
│   ├── feature_engineering.py # Feature selection and engineering
│   ├── model_trainer.py       # Training and tuning models
│   ├── evaluator.py           # Model evaluation functions
│   ├── utils.py               # Utility functions
│   ├── visualizer.py          # Visualization functions
│   └── anomaly_detector.py    # Deployment-ready anomaly detector
│
├── models/                    # Directory to save trained models
│
├── visualizations/            # Directory for generated visualizations
│
├── results/                   # Directory for prediction results
│
├── main.py                    # Main script to run the training pipeline
├── predict.py                 # Script for making predictions on new data
└── README.md                  # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/network-anomaly-detection.git
cd network-anomaly-detection
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

Place your CIC-IDS 2017 dataset files in the `data` directory. The system expects CSV files.

### Training Models

To train models, run the main script:

```bash
python main.py --data-dir ./data --output-dir ./models --target-column Label --num-features 20 --models random_forest,gradient_boosting,logistic_regression
```

Arguments:
- `--data-dir`: Directory containing the dataset files
- `--output-dir`: Directory to save models and results
- `--target-column`: Name of the target column in the dataset
- `--test-size`: Proportion of the dataset to include in the test split (default: 0.2)
- `--num-features`: Number of features to select (default: 20)
- `--models`: Comma-separated list of models to train

### Making Predictions

To make predictions on new data:

```bash
python predict.py --input ./data/test_data.csv --output ./results --model ./models/best_model.joblib --preprocessor ./models/preprocessor.joblib --feature-engineer ./models/feature_engineer.joblib
```

Arguments:
- `--input`: Path to input CSV file
- `--output`: Directory to save prediction results
- `--model`: Path to trained model file
- `--preprocessor`: Path to preprocessor file
- `--feature-engineer`: Path to feature engineer file
- `--target-column`: Name of target column (if present for evaluation)
- `--threshold`: Probability threshold for anomaly detection (default: 0.5)
- `--batch-size`: Batch size for processing large files (default: 10000)
- `--explain`: Generate explanations for predictions
- `--explain-samples`: Number of random samples to explain (default: 10)
- `--optimize-threshold`: Optimize the threshold based on evaluation metrics

## Features

### Data Processing
- Automatic detection of target column
- Handling of missing values and outliers
- Encoding of categorical features
- Feature scaling

### Feature Engineering
- Creation of rate-based features
- Forward/backward traffic ratio features
- Protocol-based feature extraction
- Statistical feature generation

### Model Training
- Support for multiple classification algorithms
- Hyperparameter optimization
- Cross-validation
- Handling of imbalanced classes
- Ensemble model creation

### Evaluation
- Comprehensive metrics (accuracy, precision, recall, F1-score, ROC AUC)
- Confusion matrix visualization
- ROC and precision-recall curves
- Feature importance analysis

### Prediction
- Batch processing for large datasets
- Threshold optimization
- Prediction explanation
- Output in CSV format

## Supported Models

- Random Forest
- Gradient Boosting
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- K-Nearest Neighbors (KNN)
- Multi-layer Perceptron (MLP)
- AdaBoost
- Ensemble (Voting Classifier of best models)

## Example Workflow

1. Prepare your data files in the `data` directory
2. Run the training pipeline:
   ```bash
   python main.py --data-dir ./data --target-column Label
   ```
3. Make predictions on new data:
   ```bash
   python predict.py --input ./data/new_traffic.csv --output ./results
   ```
4. Analyze results in the `results` directory

## Requirements

- Python 3.7+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib


## Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/network-anomaly-detection](https://github.com/yourusername/network-anomaly-detection)