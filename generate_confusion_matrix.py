import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Load the metrics JSON file
metrics_file = "C:/Users/khatr/OneDrive/Documents/InternshipProjects/Anomaly detection/anomaly-detection-in-network-traffic/results/test_data_metrics_20250329_160130.json"
with open(metrics_file, 'r') as f:
    metrics = json.load(f)

# Extract confusion matrix
confusion_matrix = np.array(metrics['confusion_matrix'])

# Get class names if available
if 'classification_report' in metrics:
    classes = list(metrics['classification_report'].keys())
    # Filter out non-class keys like 'accuracy', 'macro avg', etc.
    classes = [c for c in classes if not c.startswith('macro') and not c.startswith('weighted') and c != 'accuracy']
else:
    classes = [f"Class {i}" for i in range(len(confusion_matrix))]

# Create visualization
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()

# Save the visualization
output_path = "C:/Users/khatr/OneDrive/Documents/InternshipProjects/Anomaly detection/anomaly-detection-in-network-traffic/results/confusion_matrix.png"
plt.savefig(output_path)
print(f"Confusion matrix visualization saved to {output_path}")