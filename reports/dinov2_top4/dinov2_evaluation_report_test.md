# DINOv2 Tuberculosis Detection - Test Evaluation Report

- **Model Architecture**: `facebook/dinov2-small`
- **Evaluation Split**: `test`
- **Total Samples**: 331

## Summary Metrics

| Metric | Value |
|---|---|
| **Accuracy** | 0.5952 (59.52%) |
| **Precision (TB)** | 0.5739 |
| **Recall / Sensitivity (TB)** | 0.7857 |
| **Specificity (Normal)** | 0.3988 |
| **F1-Score** | 0.6633 |
| **ROC-AUC** | 0.6677 |

## Confusion Matrix

| | Predicted Normal | Predicted TB | Total True |
|---|---|---|---|
| **True Normal** | 65 (TN) | 98 (FP) | 163 |
| **True TB** | 36 (FN) | 132 (TP) | 168 |
| **Total Predicted** | 101 | 230 | 331 |

## Classification Breakdown

```text
              precision    recall  f1-score   support

      Normal       0.64      0.40      0.49       163
          TB       0.57      0.79      0.66       168

    accuracy                           0.60       331
   macro avg       0.61      0.59      0.58       331
weighted avg       0.61      0.60      0.58       331

```
## Visualizations

- Confusion Matrix: `figures/confusion_matrix_dinov2_test.png`
- ROC Curve: `figures/roc_curve_dinov2_test.png`
