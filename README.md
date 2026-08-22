# Twitter Emotion Classification

Fine-tuned text classification model to detect emotion/mood from tweets. Classifies text into 6 emotion categories: **sadness, fear, joy, anger, love, surprise**.

# Install dependencies
uv pip install -r requirements.txt

# Register as a Jupyter kernel (for VS Code notebooks)
uv pip install ipykernel
python -m ipykernel install --user --name=ktl_env --display-name "Python (ktl_env)"
```

## Results (3 epochs, test set = 1000 samples)

| Emotion   | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Sadness   | 0.91      | 0.96   | 0.93     | 275     |
| Fear      | 0.91      | 0.90   | 0.90     | 360     |
| Joy       | 0.73      | 0.76   | 0.74     | 71      |
| Anger     | 0.91      | 0.90   | 0.91     | 141     |
| Love      | 0.94      | 0.76   | 0.84     | 116     |
| Surprise  | 0.69      | 0.84   | 0.76     | 37      |
| **Accuracy**     |    |        | **0.89** | 1000    |
| Macro avg        | 0.85 | 0.85 | 0.85   | 1000    |
| Weighted avg     | 0.89 | 0.89 | 0.89   | 1000    |

**Overall accuracy: 89%** | **Weighted F1: 0.89**

## Confusion Matrix

<img width="637" height="540" alt="image" src="https://github.com/user-attachments/assets/bb5aadb9-df6b-4915-ad34-36e6c194e2fb" />


Key misclassification patterns:
- **Joy → Fear**: 20 samples — the largest single confusion in the matrix
- **Love → Surprise**: 12 samples, **Love → Fear**: 6 samples
- **Sadness → Fear**: 13 samples
- **Anger → Fear**: 8 samples

Fear appears to be a frequent "catch-all" mispredicted class across others — worth checking for overlapping vocabulary/tone between fear and the classes bleeding into it (joy, love, sadness, anger).

## Observations

- Strong classes: **Sadness, Fear, Anger** (F1 ≥ 0.90)
- Weak classes: **Joy** (F1 0.74) and **Surprise** (F1 0.76) — both have the lowest support (71 and 37 samples) → **class imbalance**
- Gap between macro avg (0.85) and weighted avg (0.89) confirms the imbalance is dragging down minority-class performance

## Next Steps

- [ ] Collect/augment more samples for **Joy** and **Surprise**
- [ ] Apply class weighting in the loss function (`class_weight='balanced'`)
- [ ] Investigate Joy↔Fear and Love↔Surprise confusion — check for ambiguous/overlapping tweet phrasing
- [ ] Try oversampling (SMOTE) or undersampling to balance classes
- [ ] Re-evaluate after rebalancing
