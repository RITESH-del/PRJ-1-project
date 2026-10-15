# DenseNet121 Tuberculosis Detection - Test Evaluation Report

- **Model Architecture**: `densenet121`
- **Evaluation Split**: `test`
- **Total Samples**: 331

## Summary Metrics

| Metric | Value |
|---|---|
| **Accuracy** | 0.5891 (58.91%) |
| **Precision (TB)** | 0.7051 |
| **Recall / Sensitivity (TB)** | 0.3274 |
| **Specificity (Normal)** | 0.8589 |
| **F1-Score** | 0.4472 |
| **ROC-AUC** | 0.7046 |

## Confusion Matrix

| | Predicted Normal | Predicted TB | Total True |
|---|---|---|---|
| **True Normal** | 140 (TN) | 23 (FP) | 163 |
| **True TB** | 113 (FN) | 55 (TP) | 168 |
| **Total Predicted** | 253 | 78 | 331 |

## Classification Breakdown

```text
              precision    recall  f1-score   support

      Normal       0.55      0.86      0.67       163
          TB       0.71      0.33      0.45       168

    accuracy                           0.59       331
   macro avg       0.63      0.59      0.56       331
weighted avg       0.63      0.59      0.56       331

```
## Visualizations

- Confusion Matrix: `figures/confusion_matrix_densenet121_test.png`
- ROC Curve: `figures/roc_curve_densenet121_test.png`
