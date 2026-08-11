# Student Performance — Classical ML Analysis
Applying regression and classification models to a real dataset of 1000 students, predicting exam outcomes from academic and demographic data.

## What This Project Does
- **Regression**: Predicts exact math score using reading score, writing score, test preparation status, and parental education level
- **Classification**: Predicts pass/fail (math score ≥ 50), comparing three approaches to handle class imbalance in the data (86.5% of students passed)

# Results

**Regression (Linear Regression)**
 MAE: ~7.35 points
 R²: ~0.69

**Classification — comparing three models**

Model | Accuracy | Precision (fail) | Recall (fail) 
Logistic Regression | 0.855 | 0.61 | 0.41 
Logistic Regression (balanced) | 0.81 | 0.47 | 0.82 
Random Forest (balanced) | 0.80 | 0.42 | 0.53 

## Key Findings

- Adding more features (test prep, parental education) barely improved either model — reading and writing scores already captured most of the useful signal.
- The dataset is imbalanced (most students pass), which caused every model to lean toward predicting "pass," missing many at-risk students.
- Using `class_weight="balanced"` significantly improved recall (catching more at-risk students) at the cost of precision (more false alarms) — a real, unavoidable tradeoff.
- Given the goal of identifying students who need support, the balanced Logistic Regression model is the most appropriate choice — missing an at-risk student is more costly than a false alarm.

## Tools Used
- Pandas — data loading and feature encoding
- Scikit-learn — LinearRegression, LogisticRegression, RandomForestClassifier, train/test splitting, evaluation metrics