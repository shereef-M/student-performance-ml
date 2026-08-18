"""
Student Performance - Classical ML Analysis
Predicting exam outcomes from student data using regression and classification.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("students_perf.csv")

# --- Encode categorical features ---
df["test_prep_numeric"] = df["test preparation course"].map({"completed": 1, "none": 0})
education_mapping = {
    "high school": 0, "some high school": 1, "some college": 2,
    "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5
}
df["education_numeric"] = df["parental level of education"].map(education_mapping)
df["passed"] = (df["math score"] >= 50).astype(int)

# PART 1: REGRESSION — Predicting exact math score
X = df[["reading score", "writing score", "test_prep_numeric", "education_numeric"]]
y = df["math score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
reg_predictions = reg_model.predict(X_test)

print("--- Regression Results ---")
print("MAE:", mean_absolute_error(y_test, reg_predictions))
print("R²:", r2_score(y_test, reg_predictions))


# PART 2: CLASSIFICATION — Predicting pass/fail

X = df[["reading score", "writing score"]]
y = df["passed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Classification: Plain Logistic Regression ---")
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision (fail):", precision_score(y_test, predictions, pos_label=0))
print("Recall (fail):", recall_score(y_test, predictions, pos_label=0))

print("\n--- Classification: Balanced Logistic Regression ---")
model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision (fail):", precision_score(y_test, predictions, pos_label=0))
print("Recall (fail):", recall_score(y_test, predictions, pos_label=0))

print("\n--- Classification: Random Forest (balanced) ---")
model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision (fail):", precision_score(y_test, predictions, pos_label=0))
print("Recall (fail):", recall_score(y_test, predictions, pos_label=0))

print("\n--- Classification: k-NN (distance-weighted) ---")
model = KNeighborsClassifier(n_neighbors=5, weights="distance")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision (fail):", precision_score(y_test, predictions, pos_label=0))
print("Recall (fail):", recall_score(y_test, predictions, pos_label=0))


print("\n--- Classification: SVM (balanced) ---")
model = SVC(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision (fail):", precision_score(y_test, predictions, pos_label=0))
print("Recall (fail):", recall_score(y_test, predictions, pos_label=0))

print("\n--- Classification: Naive Bayes ---")
model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision (fail):", precision_score(y_test, predictions, pos_label=0))
print("Recall (fail):", recall_score(y_test, predictions, pos_label=0))