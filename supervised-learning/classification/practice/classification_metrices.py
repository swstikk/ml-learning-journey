"""
================================================================================
  PRACTICE QUESTION PAPER: CLASSIFICATION METRICS
================================================================================
Bhai, saare challenges yahan hain. Tere liye sab data aur imports ready hain.
Har challenge ka code uske neeche block mein likh!

RULES:
1. Apna answer print karke check kar!
2. Sahi plots display kar!
3. Jo samajh na aaye, mujhse pooch!
"""

# ==========================================
# IMPORTS (Sab ready hain)
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.dummy import DummyClassifier

# ==========================================
# DATA LOADING (Breast Cancer)
# ==========================================
data = load_breast_cancer()
X, y = data.data, data.target
print(data.target_names)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a baseline model for you to use in the challenges
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(random_state=42))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------------------------
# CHALLENGE 1: Manual Confusion Matrix
# ------------------------------------------------------------------------------
print("=== CHALLENGE 1: Manual Confusion Matrix ===")
"""
TASK: 
- BINA sklearn ke manually calculate karo:
  TP = ? TN = ? FP = ? FN = ?
  (Hint: for loop lagao, har sample check karo actual vs predicted)
- Apne answers ko sklearn ke confusion_matrix se verify karo
- Dono same aane chahiye!
"""

# TERA CODE YAHAN LIKH
# tp=0
# tn=0
# fn=0
# fp=0

# for i in range(len(y_test)):
#     if y_test[i]==1:
#         if y_pred[i]==1:
#             tp+=1
#         else:
#             fn+=1
#     else:
#         if y_pred[i]==1:
#             fp+=1
#         else:
#             tn+=1

# print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
# cm= confusion_matrix(y_test,y_pred)
# print("\n sklearn confusion matrix:")
# print(cm)

# ------------------------------------------------------------------------------
# CHALLENGE 2: classification_report Padho
# ------------------------------------------------------------------------------
print("\n=== CHALLENGE 2: classification_report Padho ===")
"""
TASK: 
- `classification_report(y_test, y_pred)` print karo.
- Isko padh ke answer karo (print statements mein):
    1. Kaunsi class ka recall zyada hai?
    2. Kaunsi class ka precision zyada hai?
    3. Kya ye model cancer detection ke liye safe hai? Kyun? (Hint: cancer = class 0 in this dataset! Check `data.target_names`!)
"""

# TERA CODE YAHAN LIKH:
# print( classification_report(y_test,y_pred))
#  calss 1 ka recall jyada haii 
# class 0 ka precision jyada hai
#  hnn ye safe haii lekin real world me 99 % chahiye hota hai 


# ------------------------------------------------------------------------------
# CHALLENGE 3: ROC Curve Plot
# ------------------------------------------------------------------------------
print("\n=== CHALLENGE 3: ROC Curve Plot ===")
"""
TASK: 
- `roc_curve` use karke fpr, tpr nikalo.
- ROC curve plot karo:
    - X-axis: FPR
    - Y-axis: TPR
    - Diagonal line bhi draw karo (random classifier)
    - Title mein `roc_auc_score` value likho
- DummyClassifier (strategy="stratified") train karo aur uska bhi ROC curve same graph pe plot karo
- Kaunsa model better hai visually?
"""

# TERA CODE YAHAN LIKH:
fpr, tpr, thresholds = roc_curve(y_test, y_proba) # Use y_proba for ROC!
auc_score = roc_auc_score(y_test, y_proba)

# 1. Train Dummy Classifier
dummy = DummyClassifier(strategy="stratified", random_state=42)
dummy.fit(X_train, y_train)
y_dummy_proba = dummy.predict_proba(X_test)[:, 1]
fpr_dummy, tpr_dummy, _ = roc_curve(y_test, y_dummy_proba)
dummy_auc = roc_auc_score(y_test, y_dummy_proba)

# 2. Plot Everything
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc_score:.3f})", color="blue", linewidth=2)
plt.plot(fpr_dummy, tpr_dummy, label=f"Dummy Classifier (AUC = {dummy_auc:.3f})", color="orange", linestyle="--")
plt.plot([0, 1], [0, 1], color='red', linestyle=':', label='Random Baseline (AUC = 0.5)') # Diagonal line

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR / Recall)")
plt.title(f"ROC CURVE COMPARISON\nLogReg AUC: {auc_score:.3f}")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# ------------------------------------------------------------------------------
# CHALLENGE 4: Imbalanced Data Experiment
# ------------------------------------------------------------------------------
print("\n=== CHALLENGE 4: Imbalanced Data Experiment ===")
"""
TASK: 
- Naya imbalanced data banana hai aur 2 models test karne hain:
"""
X_imb, y_imb = make_classification(n_samples=1000, weights=[0.95, 0.05], random_state=42)
Xi_train, Xi_test, yi_train, yi_test = train_test_split(X_imb, y_imb, test_size=0.2, random_state=42)

"""
TERA CODE YAHAN LIKH:
1. Ek normal LogisticRegression train kar and predict kar.
2. Ek LogisticRegression(class_weight='balanced') train kar and predict kar.
3. Dono models ka `classification_report` print kar aur compare kar Class 1 ke results!
"""

nlr=LogisticRegression(random_state=42)
nlrb= LogisticRegression(class_weight="balanced", random_state=42 )
nlr.fit(Xi_train,yi_train)
nlrb.fit(Xi_train,yi_train)
print("normal logistic regression:")
print(classification_report(yi_test,nlr.predict(Xi_test)))
print("balanced logistic regression:")
print(classification_report(yi_test,nlrb.predict(Xi_test)))

# balance wala better hai kyu usne 55% ka recall diya jabki normal wale ne 9% kaa or hame wahi to nhi chahiye hai
#  lekin agar hame dhilaa saaa model chahiye joo jyada strict naa ho too normal wala best hai 





# ------------------------------------------------------------------------------
# CHALLENGE 5: Threshold Tuning
# ------------------------------------------------------------------------------
print("\n=== CHALLENGE 5: Threshold Tuning ===")
"""
TASK: 
- thresholds list di hui hai. Har threshold ke liye y_pred naya banao.
- Precision, Recall, aur F1 calculate karke lists mein save karo.
- Teeno ko line plot (X-axis = thresholds) par plot karo!
- Pata karo best F1 score kis threshold par mila?
"""
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
precisions = []
recalls = []
f1_scores = []

# TERA CODE YAHAN LIKH:
# Model ek baar train karna kaafi hai loop ke bahar
logr = LogisticRegression(random_state=42)
logr.fit(X_train, y_train)
y_pred = logr.predict_proba(X_test)[:, 1] # Fixed ; to :

for t in thresholds:
    y_pred_binary = [1 if i >= t else 0 for i in y_pred]
    precisions.append(precision_score(y_test, y_pred_binary))
    recalls.append(recall_score(y_test, y_pred_binary))
    f1_scores.append(f1_score(y_test, y_pred_binary))

# Ploting the metrics!
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, marker='o', label='Precision', color='blue')
plt.plot(thresholds, recalls, marker='o', label='Recall', color='red')
plt.plot(thresholds, f1_scores, marker='s', label='F1 Score', color='green', linewidth=2)
plt.title("Precision, Recall, and F1 Score vs. Threshold")
plt.xlabel("Threshold (Probability cut-off)")
plt.ylabel("Score")
plt.xticks(thresholds)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# answer= 0.8


