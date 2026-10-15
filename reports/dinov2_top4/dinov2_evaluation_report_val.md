# DINOv2 Tuberculosis Detection - Val Evaluation Report

- **Model Architecture**: `facebook/dinov2-small`
- **Evaluation Split**: `val`
- **Total Samples**: 331

## Summary Metrics

| Metric | Value |
|---|---|
| **Accuracy** | 0.6375 (63.75%) |
| **Precision (TB)** | 0.6062 |
| **Recall / Sensitivity (TB)** | 0.8155 |
| **Specificity (Normal)** | 0.4540 |
| **F1-Score** | 0.6954 |
| **ROC-AUC** | 0.7206 |

## Confusion Matrix

| | Predicted Normal | Predicted TB | Total True |
|---|---|---|---|
| **True Normal** | 74 (TN) | 89 (FP) | 163 |
| **True TB** | 31 (FN) | 137 (TP) | 168 |
| **Total Predicted** | 105 | 226 | 331 |

## Classification Breakdown

```text
              precision    recall  f1-score   support

      Normal       0.70      0.45      0.55       163
          TB       0.61      0.82      0.70       168

    accuracy                           0.64       331
   macro avg       0.66      0.63      0.62       331
weighted avg       0.65      0.64      0.62       331

```
## Visualizations

- Confusion Matrix: `figures/confusion_matrix_dinov2_val.png`
- ROC Curve: `figures/roc_curve_dinov2_val.png`
- Training Curves: `figures/training_curves_dinov2.png`
