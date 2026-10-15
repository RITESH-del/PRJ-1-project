# DINOv2 Tuberculosis Detection - Test Evaluation Report

- **Model Architecture**: `facebook/dinov2-small`
- **Evaluation Split**: `test`
- **Total Samples**: 331

## Summary Metrics

| Metric | Value |
|---|---|
| **Accuracy** | 0.6254 (62.54%) |
| **Precision (TB)** | 0.6897 |
| **Recall / Sensitivity (TB)** | 0.4762 |
| **Specificity (Normal)** | 0.7791 |
| **F1-Score** | 0.5634 |
| **ROC-AUC** | 0.7140 |

## Confusion Matrix

| | Predicted Normal | Predicted TB | Total True |
|---|---|---|---|
| **True Normal** | 127 (TN) | 36 (FP) | 163 |
| **True TB** | 88 (FN) | 80 (TP) | 168 |
| **Total Predicted** | 215 | 116 | 331 |

## Classification Breakdown

```text
              precision    recall  f1-score   support

      Normal       0.59      0.78      0.67       163
          TB       0.69      0.48      0.56       168

    accuracy                           0.63       331
   macro avg       0.64      0.63      0.62       331
weighted avg       0.64      0.63      0.62       331

```
## Visualizations

- Confusion Matrix: `figures/confusion_matrix_dinov2_test.png`
- ROC Curve: `figures/roc_curve_dinov2_test.png`
