# Student Performance — Classical ML Analysis

Applying regression, classification, and clustering models to a real dataset of 1000 students, predicting exam outcomes from academic and demographic data.

## What This Project Does

- **Regression**: Predicts exact math score using reading score, writing score, test preparation status, and parental education level
- **Classification**: Predicts pass/fail (math score ≥ 50), comparing six different models
- **Clustering**: Groups students by score similarity alone (no labels used) to see if natural groupings emerge
- **Cross-validation**: Tests whether classification results are stable across different data splits

## Results

**Regression (Linear Regression)**
- MAE: ~7.35 points
- R²: ~0.69

**Classification — comparing six models (recall/precision measured on the "failed" class)**

| Model | Accuracy | Precision (fail) | Recall (fail) |
| Logistic Regression | 0.855 | 0.61 | 0.41 |
| Logistic Regression (balanced) | 0.81 | 0.47 | 0.82 |
| Random Forest (balanced) | 0.80 | 0.42 | 0.53 |
| k-NN (distance-weighted) | 0.83 | 0.50 | 0.47 |
| SVM (balanced) | 0.785 | 0.43 | 0.88 |
| Naive Bayes | 0.84 | 0.52 | 0.65 |

**Clustering (k-means, 3 groups, unsupervised)**

| Cluster | Avg Reading | Avg Writing | Pass Rate |
| 0 (high) | 85.2 | 84.6 | 100% |
| 1 (middle) | 68.4 | 67.7 | 95% |
| 2 (low) | 50.5 | 48.1 | 55% |

**Cross-validation (5-fold, plain Logistic Regression)**
- Recall per fold ranged from 0.30 to 0.63
- Average recall: ~0.44

## Key Findings

- Adding more features (test prep, parental education) barely improved either model — reading and writing scores already captured most of the useful signal.
- The dataset is imbalanced (86.5% of students passed), which caused every model to lean toward predicting "pass," missing many at-risk students.
- Different techniques handle this imbalance differently: `class_weight="balanced"` and distance-weighted k-NN both improved recall, but with different precision tradeoffs — distance weighting gave a better recall gain without hurting precision.
- SVM achieved the highest recall (0.88) of any model tested, at the cost of the lowest accuracy — illustrating that no single model is "best" without considering the actual use case.
- k-means clustering, run with zero knowledge of pass/fail, still discovered groups that closely tracked real outcomes (100%/95%/55% pass rates) — showing that "found structure" in unsupervised learning can align meaningfully with real-world categories, even without being told what to look for.
- Cross-validation revealed that a single train/test split can be misleading: recall varied from 0.30 to 0.63 depending on the split, despite averaging 0.44 — a reminder that one evaluation number should not be fully trusted on its own.

## Tools Used

- Pandas — data loading and feature encoding
- Scikit-learn — LinearRegression, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier, SVC, GaussianNB, KMeans, cross_val_score