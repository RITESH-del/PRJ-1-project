# DINOv2 Tuberculosis Detection - Test Evaluation Report

- **Model Architecture**: `facebook/dinov2-small`
- **Evaluation Split**: `test`
- **Total Samples**: 331

## Summary Metrics

| Metric | Value |
|---|---|
| **Accuracy** | 0.7674 (76.74%) |
| **Precision (TB)** | 0.7630 |
| **Recall / Sensitivity (TB)** | 0.7857 |
| **Specificity (Normal)** | 0.7485 |
| **F1-Score** | 0.7742 |
| **ROC-AUC** | 0.8307 |

## Confusion Matrix

| | Predicted Normal | Predicted TB | Total True |
|---|---|---|---|
| **True Normal** | 122 (TN) | 41 (FP) | 163 |
| **True TB** | 36 (FN) | 132 (TP) | 168 |
| **Total Predicted** | 158 | 173 | 331 |

## Classification Breakdown

```text
              precision    recall  f1-score   support

      Normal       0.77      0.75      0.76       163
          TB       0.76      0.79      0.77       168

    accuracy                           0.77       331
   macro avg       0.77      0.77      0.77       331
weighted avg       0.77      0.77      0.77       331

```
## Visualizations

- Confusion Matrix: `figures/confusion_matrix_dinov2_test.png`
- ROC Curve: `figures/roc_curve_dinov2_test.png`
