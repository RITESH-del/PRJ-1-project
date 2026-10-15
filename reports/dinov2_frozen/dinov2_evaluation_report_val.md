# DINOv2 Tuberculosis Detection - Val Evaluation Report

- **Model Architecture**: `facebook/dinov2-small`
- **Evaluation Split**: `val`
- **Total Samples**: 331

## Summary Metrics

| Metric | Value |
|---|---|
| **Accuracy** | 0.6556 (65.56%) |
| **Precision (TB)** | 0.7596 |
| **Recall / Sensitivity (TB)** | 0.4702 |
| **Specificity (Normal)** | 0.8466 |
| **F1-Score** | 0.5809 |
| **ROC-AUC** | 0.7331 |

## Confusion Matrix

| | Predicted Normal | Predicted TB | Total True |
|---|---|---|---|
| **True Normal** | 138 (TN) | 25 (FP) | 163 |
| **True TB** | 89 (FN) | 79 (TP) | 168 |
| **Total Predicted** | 227 | 104 | 331 |

## Classification Breakdown

```text
              precision    recall  f1-score   support

      Normal       0.61      0.85      0.71       163
          TB       0.76      0.47      0.58       168

    accuracy                           0.66       331
   macro avg       0.68      0.66      0.64       331
weighted avg       0.68      0.66      0.64       331

```
## Visualizations

- Confusion Matrix: `figures/confusion_matrix_dinov2_val.png`
- ROC Curve: `figures/roc_curve_dinov2_val.png`
- Training Curves: `figures/training_curves_dinov2.png`
