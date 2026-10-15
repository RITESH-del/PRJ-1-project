# DINOv2 Tuberculosis Detection - Val Evaluation Report

- **Model Architecture**: `facebook/dinov2-small`
- **Evaluation Split**: `val`
- **Total Samples**: 331

## Summary Metrics

| Metric | Value |
|---|---|
| **Accuracy** | 0.6647 (66.47%) |
| **Precision (TB)** | 0.6647 |
| **Recall / Sensitivity (TB)** | 0.6845 |
| **Specificity (Normal)** | 0.6442 |
| **F1-Score** | 0.6745 |
| **ROC-AUC** | 0.7461 |

## Confusion Matrix

| | Predicted Normal | Predicted TB | Total True |
|---|---|---|---|
| **True Normal** | 105 (TN) | 58 (FP) | 163 |
| **True TB** | 53 (FN) | 115 (TP) | 168 |
| **Total Predicted** | 158 | 173 | 331 |

## Classification Breakdown

```text
              precision    recall  f1-score   support

      Normal       0.66      0.64      0.65       163
          TB       0.66      0.68      0.67       168

    accuracy                           0.66       331
   macro avg       0.66      0.66      0.66       331
weighted avg       0.66      0.66      0.66       331

```
## Visualizations

- Confusion Matrix: `figures/confusion_matrix_dinov2_val.png`
- ROC Curve: `figures/roc_curve_dinov2_val.png`
- Training Curves: `figures/training_curves_dinov2.png`
