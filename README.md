# 🔍 Network Traffic Anomaly Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/scikit--learn-1.0+-green.svg" alt="Scikit-learn Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status">
</p>

<p align="center">
  <i>A machine learning-based system for detecting and classifying network traffic anomalies and cyber attacks</i>
</p>

## ✨ Overview

This project implements a sophisticated machine learning pipeline for network traffic anomaly detection using the CIC-IDS 2017 dataset. The system can identify various types of cyber attacks including DDoS, DoS, port scanning, brute force attempts, and web attacks by analyzing network flow features.

## 🚀 Key Features

- **📊 Comprehensive Data Pipeline**: Automated preprocessing, feature engineering, and model training
- **⚙️ Advanced Feature Engineering**: Creates 60+ engineered features from raw network traffic data
- **🤖 Multiple ML Models**: Trains and evaluates various classification algorithms
- **⚡ Performance Optimization**: Implements timeouts, memory monitoring, and efficient sampling
- **🔄 Real-time Prediction**: Detects anomalies in new network traffic data
- **📈 Detailed Evaluation**: Generates comprehensive performance metrics and visualizations

## 🏗️ Technical Architecture

The system is organized into several specialized components:

| Component | Description |
|-----------|-------------|
| **Data Loader** | Handles dataset loading, merging, and initial inspection |
| **Preprocessor** | Cleans data, handles missing values, and encodes categorical features |
| **Feature Engineer** | Creates new features and selects the most relevant ones |
| **Model Trainer** | Trains multiple models with hyperparameter tuning |
| **Evaluator** | Calculates performance metrics and creates visualizations |
| **Anomaly Detector** | Uses trained models to identify anomalies in new data |
| **Visualizer** | Creates data and result visualizations |

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/network-anomaly-detection.git
cd network-anomaly-detection

# Set up environment
pip install -r requirements.txt
```

## 📋 Usage

### Training Models

```bash
python main.py --data-dir ./data --output-dir ./models --models random_forest,logistic_regression,decision_tree
```

### Making Predictions

```bash
python predict.py --input ./data/test_data.csv --output ./results --model ./models/best_model.joblib --preprocessor ./models/preprocessor.joblib --feature-engineer ./models/feature_engineer.joblib
```

## 📊 Dataset

This project uses the CIC-IDS 2017 dataset, which contains labeled network traffic including:

- **DoS Attacks**: Hulk, GoldenEye, Slowloris, Slowhttptest
- **DDoS Attacks**: Various distributed attack patterns
- **Web Attacks**: XSS, SQL Injection, Brute Force
- **Infiltration**: Simulated insider threats
- **Brute Force**: FTP and SSH login attempts
- **Port Scanning**: Network reconnaissance activities
- **Botnet**: Command and control traffic

## 📈 Results

The model achieves excellent performance metrics on test data:

<p align="center">
  <table>
    <tr>
      <th>Metric</th>
      <th>Value</th>
    </tr>
    <tr>
      <td>Accuracy</td>
      <td>99.2%</td>
    </tr>
    <tr>
      <td>Precision</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td>Recall</td>
      <td>99.2%</td>
    </tr>
    <tr>
      <td>F1 Score</td>
      <td>99.6%</td>
    </tr>
  </table>
</p>

<p align="center">
  <img src="path/to/confusion_matrix.png" alt="Confusion Matrix" width="600">
</p>

## 🔮 Future Enhancements

- [ ] Interactive dashboard for real-time monitoring
- [ ] Support for streaming data processing
- [ ] Explainable AI techniques for better attack attribution
- [ ] Benchmark against commercial IDS solutions
- [ ] Integration with threat intelligence platforms

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📬 Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/network-anomaly-detection](https://github.com/yourusername/network-anomaly-detection)

## 🙏 Acknowledgements

- [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/) for the CIC-IDS 2017 dataset
- [Scikit-learn](https://scikit-learn.org/) for machine learning tools
- [Pandas](https://pandas.pydata.org/) for data manipulation

---

<p align="center">
  <i>Made with ❤️ for network security</i>
</p>
