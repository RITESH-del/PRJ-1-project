# DenseNet121 Tuberculosis Detection - Val Evaluation Report

- **Model Architecture**: `densenet121`
- **Evaluation Split**: `val`
- **Total Samples**: 331

## Summary Metrics

| Metric | Value |
|---|---|
| **Accuracy** | 0.7462 (74.62%) |
| **Precision (TB)** | 0.7100 |
| **Recall / Sensitivity (TB)** | 0.8452 |
| **Specificity (Normal)** | 0.6442 |
| **F1-Score** | 0.7717 |
| **ROC-AUC** | 0.8164 |

## Confusion Matrix

| | Predicted Normal | Predicted TB | Total True |
|---|---|---|---|
| **True Normal** | 105 (TN) | 58 (FP) | 163 |
| **True TB** | 26 (FN) | 142 (TP) | 168 |
| **Total Predicted** | 131 | 200 | 331 |

## Classification Breakdown

```text
              precision    recall  f1-score   support

      Normal       0.80      0.64      0.71       163
          TB       0.71      0.85      0.77       168

    accuracy                           0.75       331
   macro avg       0.76      0.74      0.74       331
weighted avg       0.76      0.75      0.74       331

```
## Visualizations

- Confusion Matrix: `figures/confusion_matrix_densenet121_val.png`
- ROC Curve: `figures/roc_curve_densenet121_val.png`
- Training Curves: `figures/training_curves_densenet121.png`
