# DenseNet121 Tuberculosis Detection - Test Evaluation Report

- **Model Architecture**: `densenet121`
- **Evaluation Split**: `test`
- **Total Samples**: 331

## Summary Metrics

| Metric | Value |
|---|---|
| **Accuracy** | 0.6677 (66.77%) |
| **Precision (TB)** | 0.6465 |
| **Recall / Sensitivity (TB)** | 0.7619 |
| **Specificity (Normal)** | 0.5706 |
| **F1-Score** | 0.6995 |
| **ROC-AUC** | 0.7214 |

## Confusion Matrix

| | Predicted Normal | Predicted TB | Total True |
|---|---|---|---|
| **True Normal** | 93 (TN) | 70 (FP) | 163 |
| **True TB** | 40 (FN) | 128 (TP) | 168 |
| **Total Predicted** | 133 | 198 | 331 |

## Classification Breakdown

```text
              precision    recall  f1-score   support

      Normal       0.70      0.57      0.63       163
          TB       0.65      0.76      0.70       168

    accuracy                           0.67       331
   macro avg       0.67      0.67      0.66       331
weighted avg       0.67      0.67      0.66       331

```
## Visualizations

- Confusion Matrix: `figures/confusion_matrix_densenet121_test.png`
- ROC Curve: `figures/roc_curve_densenet121_test.png`
