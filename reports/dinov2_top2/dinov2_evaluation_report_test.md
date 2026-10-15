# DINOv2 Tuberculosis Detection - Test Evaluation Report

- **Model Architecture**: `facebook/dinov2-small`
- **Evaluation Split**: `test`
- **Total Samples**: 331

## Summary Metrics

| Metric | Value |
|---|---|
| **Accuracy** | 0.6284 (62.84%) |
| **Precision (TB)** | 0.6331 |
| **Recall / Sensitivity (TB)** | 0.6369 |
| **Specificity (Normal)** | 0.6196 |
| **F1-Score** | 0.6350 |
| **ROC-AUC** | 0.6979 |

## Confusion Matrix

| | Predicted Normal | Predicted TB | Total True |
|---|---|---|---|
| **True Normal** | 101 (TN) | 62 (FP) | 163 |
| **True TB** | 61 (FN) | 107 (TP) | 168 |
| **Total Predicted** | 162 | 169 | 331 |

## Classification Breakdown

```text
              precision    recall  f1-score   support

      Normal       0.62      0.62      0.62       163
          TB       0.63      0.64      0.64       168

    accuracy                           0.63       331
   macro avg       0.63      0.63      0.63       331
weighted avg       0.63      0.63      0.63       331

```
## Visualizations

- Confusion Matrix: `figures/confusion_matrix_dinov2_test.png`
- ROC Curve: `figures/roc_curve_dinov2_test.png`
