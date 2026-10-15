# DenseNet121 Tuberculosis Detection - Val Evaluation Report

- **Model Architecture**: `densenet121`
- **Evaluation Split**: `val`
- **Total Samples**: 331

## Summary Metrics

| Metric | Value |
|---|---|
| **Accuracy** | 0.6465 (64.65%) |
| **Precision (TB)** | 0.8493 |
| **Recall / Sensitivity (TB)** | 0.3690 |
| **Specificity (Normal)** | 0.9325 |
| **F1-Score** | 0.5145 |
| **ROC-AUC** | 0.7558 |

## Confusion Matrix

| | Predicted Normal | Predicted TB | Total True |
|---|---|---|---|
| **True Normal** | 152 (TN) | 11 (FP) | 163 |
| **True TB** | 106 (FN) | 62 (TP) | 168 |
| **Total Predicted** | 258 | 73 | 331 |

## Classification Breakdown

```text
              precision    recall  f1-score   support

      Normal       0.59      0.93      0.72       163
          TB       0.85      0.37      0.51       168

    accuracy                           0.65       331
   macro avg       0.72      0.65      0.62       331
weighted avg       0.72      0.65      0.62       331

```
## Visualizations

- Confusion Matrix: `figures/confusion_matrix_densenet121_val.png`
- ROC Curve: `figures/roc_curve_densenet121_val.png`
