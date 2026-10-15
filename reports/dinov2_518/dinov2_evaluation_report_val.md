# DINOv2 Tuberculosis Detection - Val Evaluation Report

- **Model Architecture**: `facebook/dinov2-small`
- **Evaluation Split**: `val`
- **Total Samples**: 331

## Summary Metrics

| Metric | Value |
|---|---|
| **Accuracy** | 0.8036 (80.36%) |
| **Precision (TB)** | 0.8366 |
| **Recall / Sensitivity (TB)** | 0.7619 |
| **Specificity (Normal)** | 0.8466 |
| **F1-Score** | 0.7975 |
| **ROC-AUC** | 0.8666 |

## Confusion Matrix

| | Predicted Normal | Predicted TB | Total True |
|---|---|---|---|
| **True Normal** | 138 (TN) | 25 (FP) | 163 |
| **True TB** | 40 (FN) | 128 (TP) | 168 |
| **Total Predicted** | 178 | 153 | 331 |

## Classification Breakdown

```text
              precision    recall  f1-score   support

      Normal       0.78      0.85      0.81       163
          TB       0.84      0.76      0.80       168

    accuracy                           0.80       331
   macro avg       0.81      0.80      0.80       331
weighted avg       0.81      0.80      0.80       331

```
## Visualizations

- Confusion Matrix: `figures/confusion_matrix_dinov2_val.png`
- ROC Curve: `figures/roc_curve_dinov2_val.png`
- Training Curves: `figures/training_curves_dinov2.png`
