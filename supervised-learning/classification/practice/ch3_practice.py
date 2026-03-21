"""
================================================================================
  PRACTICE QUESTION PAPER: DECISION TREES
================================================================================
Rules:
  1. Theory answers → print() se output karo
  2. Code challenges → ek ek function / block mein likho
  3. Main niche run karke dekh skein
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score 
from sklearn.metrics import classification_report

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
target_names  = data.target_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================================================================
# SECTION A — THEORY (Short answers, print karo)
# ==============================================================================

# Q1. 10 samples: 7 Cancer, 3 Healthy. Gini impurity calculate karo.
print("=== A-Q1: Gini ===")
# TERA CODE YAHAN:
#  1- (0.49+0.09)= 0.52


# Q2. max_depth=None kyun dangerous hai?
print("\n=== A-Q2: max_depth=None ===")
# TERA CODE YAHAN: (print ek line answer)
#  overfit hojata hai

# Q3. Decision Tree ko StandardScaler lagana zaruri hai? Kyun / Kyun nahin?
print("\n=== A-Q3: Scaling ===")
# TERA CODE YAHAN:
# nhi ,value level matter nhi karta hai 

# Q4. feature_importances_ = [0.7, 0.2, 0.1] — kaunsa feature sabse important?
#     Ye decide kaise hua?
print("\n=== A-Q4: Feature Importance ===")
# TERA CODE YAHAN:
#  pahla waala 

# ==============================================================================
# SECTION B — CODING CHALLENGES
# ==============================================================================

# ─── Challenge 1: Gini From Scratch ──────────────────────────────────────────



print("\n=== B-C1: Gini From Scratch ===")
"""
- Ek function gini_impurity(labels) likho
- Test: gini([1,1,1,1]) = 0.0 | gini([1,1,0,0]) = 0.5 | gini([1,1,1,1,0]) ≈ 0.32
"""
# TERA CODE YAHAN:
def gini_impurity(labels):
    if len(labels) == 0: return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return 1 - np.sum(probs**2)


# ─── Challenge 2: Overfitting Depth Experiment ───────────────────────────────
print("\n=== B-C2: Depth vs Accuracy ===")
"""
- max_depth = 1..20 tak loop karo
- Har depth pe Train + Test accuracy plot karo
- Print: "Overfitting starts at depth = ___"
"""
train_acc = []
test_acc = []
depths = range(1, 21)

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    train_acc.append(clf.score(X_train, y_train))
    test_acc.append(clf.score(X_test, y_test))

plt.figure(figsize=(10, 5))
plt.plot(depths, train_acc, label="Train Accuracy", marker="o")
plt.plot(depths, test_acc, label="Test Accuracy", marker="s")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting in Decision Trees (Depth vs Accuracy)")
plt.xticks(depths)
plt.legend()
plt.grid(True, alpha=0.3)
# plt.show()  # Commented out so it doesn't block Challenge 3

print("Bhai, Overfitting starts clearly after depth = 4 ! (Test accuracy ruk jaati hai, par Train accuracy 100% tak bhaagti hai)")


# ─── Challenge 3: Tree Visualization ─────────────────────────────────────────
print("\n=== B-C3: Tree Visualization ===")
"""
- max_depth=3 ka tree train karo
- plot_tree() se visualize (plt.show())
- Print: depth, leaf nodes, root feature name
  Hint: model.get_depth(), model.get_n_leaves(), model.tree_.feature[0]
"""
dt3 = DecisionTreeClassifier(max_depth=3, random_state=42)
dt3.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(dt3, feature_names=feature_names, class_names=target_names, filled=True, rounded=True)
plt.title("Decision Tree Visualization (max_depth=3)")
# plt.show() # Commented out so it doesn't block the next running steps

print(f"Tree ki Depth: {dt3.get_depth()}")
print(f"Leaf Nodes (Total final answers): {dt3.get_n_leaves()}")
root_feat_idx = dt3.tree_.feature[0]
print(f"Root Node pe sabse pehla question kispe pucha? : {feature_names[root_feat_idx]} (Id: {root_feat_idx})")


# ─── Challenge 4: Feature Importance Bar Chart ───────────────────────────────
print("\n=== B-C4: Feature Importance ===")
"""
- Best depth use karo (Challenge 2 se)
- Top 10 features ka HORIZONTAL bar chart banao
- Print: "Top 3 features: ___"
"""
best_dt = DecisionTreeClassifier(max_depth=4, random_state=42)
best_dt.fit(X_train, y_train)
importances = best_dt.feature_importances_

# Get indices of top 10 features
indices = np.argsort(importances)[::-1][:10]
top_features = [feature_names[i] for i in indices]
top_importances = importances[indices]

# Plot (reversed so highest is at the top of the horizontal bar)
plt.figure(figsize=(10, 6))
plt.barh(top_features[::-1], top_importances[::-1], color='green')
plt.title("Top 10 Feature Importances (max_depth=4)")
plt.xlabel("Gini Importance")
plt.show()

print(f"Top 3 features ye hain: {top_features[0]}, {top_features[1]}, {top_features[2]}")


# ─── Challenge 5: GridSearchCV — Best Tree ───────────────────────────────────
print("\n=== B-C5: GridSearchCV ===")
"""
param_grid = {
    'max_depth': [2, 3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf':  [1, 2, 5, 10],
}
- GridSearchCV lagao (cv=5, scoring='accuracy')
- Print: Best params, CV score, Test score
- Best model ka classification_report print karo
"""
param_grid = {
    'max_depth': [2, 3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf':  [1, 2, 5, 10],
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print(f"Sabse jabardast parameters: {grid.best_params_}")
print(f"Best CV Score (Train mein): {grid.best_score_:.4f}")
print(f"Test Score (Naye data par): {best_model.score(X_test, y_test):.4f}")

print("\nBest Model ka Classification Report:")
print(classification_report(y_test, best_model.predict(X_test)))


# ==============================================================================
