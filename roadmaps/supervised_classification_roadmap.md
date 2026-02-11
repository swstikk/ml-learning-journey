# Supervised Learning: CLASSIFICATION - Complete Roadmap

> **Structure:** Chapter → Theory → Code → Questions → Next Chapter
> **Prerequisites:** Regression DONE ✅ (Ch 1-8), Scaling + Pipeline (do before starting)
> **Status:** NOT STARTED

---

## 📋 Pre-Classification: Quick Bridge (Before Starting)

> **Do these FIRST — most already partially covered**

### Bridge 1: Scaling + Pipelines (half day)
- [ ] StandardScaler, MinMaxScaler, RobustScaler — when to use which
- [ ] ColumnTransformer (different transforms for different columns)
- [ ] Full production pipeline: Select → Encode → Scale → Model
- [ ] **Code Practice**

### Bridge 2: Categorical Encoding (needed for classification datasets!)
- [ ] `pd.get_dummies()` vs `OneHotEncoder`
- [ ] Label Encoding (ordinal categories)
- [ ] When to use which

---

## 🗺️ CLASSIFICATION OVERVIEW

```
Chapters:
├── Ch 1: Logistic Regression          ← Start here (connects to Linear Regression!)
├── Ch 2: Classification Metrics       ← Right after Logistic (you need metrics to evaluate)
├── Ch 3: Decision Trees               ← Non-linear, visual, easy
├── Ch 4: Random Forest                ← Ensemble of trees
├── Ch 5: SVM (Support Vector Machine) ← Math-heavy, powerful
├── Ch 6: KNN (K-Nearest Neighbors)    ← Simplest concept
├── Ch 7: Naive Bayes                  ← Probability-based
├── Ch 8: Model Comparison Project     ← All models on one dataset
└── Ch 9: Ensemble Methods (Boosting)  ← XGBoost, Gradient Boosting

Total Chapters: 9
```

---

## Chapter 1: Logistic Regression

> **Ye regression se directly connected hai! Linear Regression + Sigmoid = Logistic Regression**

### Theory:
- [ ] 1.1 What is Logistic Regression? (Classification, not regression!)
- [ ] 1.2 **Sigmoid Function**: σ(z) = 1/(1 + e^(-z))
  - Maps any number to [0, 1] → Probability!
  - z = w₀ + w₁x₁ + w₂x₂ + ... (same as linear regression!)
- [ ] 1.3 **Decision Boundary**: If P > 0.5 → Class 1, else Class 0
- [ ] 1.4 **Loss Function**: Log Loss (Cross-Entropy)
  - NOT MSE! (MSE creates non-convex problem)
  - Log Loss = -[y·log(p) + (1-y)·log(1-p)]
- [ ] 1.5 **Multiclass**: One-vs-Rest (OvR) vs Multinomial
- [ ] 1.6 **Regularization**: C parameter (C = 1/λ)
  - Small C = strong regularization (like high alpha in Ridge!)
  - Large C = weak regularization
- [ ] 1.7 **predict_proba()**: Get actual probabilities, not just 0/1

### Code Practice:
- [ ] **Create:** `logistic_regression.py`
  - Breast Cancer dataset (sklearn) — binary classification
  - Train LogisticRegression
  - Print coefficients + interpretation
  - predict vs predict_proba comparison
  - Sigmoid function visualize from scratch
  - Test different C values

### Questions (10):
- [ ] Complete all 10 conceptual + coding questions

### Connection to Regression:
```
Linear Regression:    y = w₀ + w₁x₁ + w₂x₂      → Number (continuous)
Logistic Regression:  P = σ(w₀ + w₁x₁ + w₂x₂)   → Probability [0,1]
                           ↑
                      Same linear equation, just wrapped in sigmoid!
```

---

## Chapter 2: Classification Metrics

> **MUST learn right after Logistic — you need these to evaluate ALL models**

### Theory:
- [ ] 2.1 **Confusion Matrix** (Foundation of everything!)
  ```
                Predicted
               0       1
  Actual  0  [ TN  |  FP ]
          1  [ FN  |  TP ]
  ```
- [ ] 2.2 **Accuracy** = (TP + TN) / Total
  - Problem: Misleading for imbalanced data! (99% negative → predict all 0 = 99% accuracy!)
- [ ] 2.3 **Precision** = TP / (TP + FP) — "Of predicted positives, how many correct?"
  - Use when: False Positives costly (spam filter — don't want good email in spam)
- [ ] 2.4 **Recall** = TP / (TP + FN) — "Of actual positives, how many found?"
  - Use when: False Negatives costly (cancer detection — don't miss cancer!)
- [ ] 2.5 **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
  - Balance between Precision and Recall
- [ ] 2.6 **ROC Curve & AUC**
  - ROC: TPR vs FPR at different thresholds
  - AUC: Area under ROC (0.5 = random, 1.0 = perfect)
- [ ] 2.7 **When to Use Which Metric** (Decision Matrix)
- [ ] 2.8 **Multi-class**: Macro vs Weighted vs Micro averaging

### Code Practice:
- [ ] **Create:** `classification_metrics.py`
  - Manually calculate TP, TN, FP, FN
  - Precision, Recall, F1 from scratch
  - sklearn: `classification_report`, `confusion_matrix`
  - Plot ROC curve + calculate AUC
  - Compare metrics on imbalanced dataset

### Questions (10):
- [ ] Complete all 10 questions

---

## Chapter 3: Decision Trees

> **Visual, intuitive — tree of YES/NO questions**

### Theory:
- [ ] 3.1 **Tree Structure**: Root → Internal Nodes → Leaf Nodes
- [ ] 3.2 **How Splits Decided**:
  - Gini Impurity: 1 - Σ(pᵢ²) → 0 = pure, 0.5 = max impurity
  - Entropy: -Σ pᵢ × log₂(pᵢ) → 0 = pure
  - Information Gain = Entropy(parent) - weighted Entropy(children)
- [ ] 3.3 **Overfitting Control** (Hyperparameters):
  - max_depth, min_samples_split, min_samples_leaf, max_features
- [ ] 3.4 **Advantages**: No scaling needed, handles categorical, interpretable
- [ ] 3.5 **Disadvantages**: Overfits easily, unstable
- [ ] 3.6 **Decision Tree Regressor** (already used in regression context)
- [ ] 3.7 **Feature Importance** from tree: `model.feature_importances_`

### Code Practice:
- [ ] **Create:** `decision_trees.py`
  - Titanic dataset
  - Visualize tree with `plot_tree()`
  - Experiment with max_depth (2, 5, 10, None)
  - Compare train vs test score at each depth → overfitting demo!
  - Feature importance bar chart
  - GridSearchCV to tune hyperparameters

### Questions (10):
- [ ] Complete all 10 questions

---

## Chapter 4: Random Forest

> **Many trees voting = better than one tree. Ensemble method!**

### Theory:
- [ ] 4.1 **Why Ensemble?**: Single tree = overfit, Many trees = stable
- [ ] 4.2 **Bagging** (Bootstrap Aggregating):
  - Each tree trained on random subset (with replacement)
  - ~37% data left out = OOB (Out-of-Bag) → free validation!
- [ ] 4.3 **Feature Randomness**: Each split considers random subset of features
- [ ] 4.4 **Aggregation**: Classification = majority vote, Regression = average
- [ ] 4.5 **Key Hyperparameters**:
  - n_estimators (100-1000), max_depth, max_features, min_samples_leaf
- [ ] 4.6 **OOB Score**: `oob_score=True` → no need for separate val set!
- [ ] 4.7 **Advantages**: Resistant to overfitting, no scaling, feature importance
- [ ] 4.8 **Disadvantages**: Less interpretable, slower

### Code Practice:
- [ ] **Create:** `random_forest.py`
  - Compare: Single tree vs Forest (accuracy, stability)
  - Tune n_estimators (10, 50, 100, 500) → plot score vs n_estimators
  - Feature importance comparison (tree vs forest)
  - OOB score vs CV score comparison
  - GridSearchCV on multiple hyperparameters

### Questions (10):
- [ ] Complete all 10 questions

### Connection:
```
Decision Tree:  1 tree, overfits, fast
Random Forest:  100+ trees, stable, slower
Boosting:       Trees learn from each other's mistakes (Ch 9)
```

---

## Chapter 5: Support Vector Machines (SVM)

> **Math-heavy but powerful! Finds maximum margin hyperplane**

### Theory:
- [ ] 5.1 **Intuition**: Find line with MAXIMUM margin between classes
- [ ] 5.2 **Support Vectors**: Points closest to decision boundary
- [ ] 5.3 **Hard vs Soft Margin**:
  - Hard: Perfect separation (fails on noisy data)
  - Soft: Allow some misclassification (C parameter)
- [ ] 5.4 **Kernel Trick** (IMPORTANT!):
  - Linear: No transformation
  - RBF (Radial Basis Function): Most common, handles non-linear
  - Polynomial: Maps to polynomial space
  - Idea: Map to higher dimension where data IS separable!
- [ ] 5.5 **Hyperparameters**: C (regularization), gamma (RBF reach), kernel type
- [ ] 5.6 **SCALING IS CRITICAL** — SVM very sensitive to feature scale!
- [ ] 5.7 **Pros**: Good in high dimensions, memory efficient
- [ ] 5.8 **Cons**: Slow on large data, hard to interpret

### Code Practice:
- [ ] **Create:** `svm_classification.py`
  - Pipeline: StandardScaler + SVC (scaling mandatory!)
  - Compare kernels: linear, rbf, poly
  - Visualize decision boundaries (2D data)
  - GridSearchCV for C + gamma + kernel
  - Speed comparison with Random Forest

### Questions (10):
- [ ] Complete all 10 questions

---

## Chapter 6: K-Nearest Neighbors (KNN)

> **Simplest concept: "You are the average of your neighbors"**

### Theory:
- [ ] 6.1 **Algorithm**: Find K nearest points → majority vote
- [ ] 6.2 **Distance Metrics**: Euclidean, Manhattan, Minkowski
- [ ] 6.3 **Choosing K**:
  - Small K (1-3): Overfits, noisy
  - Large K (20+): Underfits, too smooth
  - Odd K for binary (avoid ties)
  - Use CV to find best K
- [ ] 6.4 **Weighted KNN**: Closer neighbors = more influence (`weights='distance'`)
- [ ] 6.5 **SCALING IS CRITICAL** — Distance-based!
- [ ] 6.6 **Lazy Learning**: No training! Just stores data
- [ ] 6.7 **Curse of Dimensionality**: Struggles with many features
- [ ] 6.8 **Pros**: Simple, no training, non-parametric
- [ ] 6.9 **Cons**: Slow prediction, memory heavy, needs scaling

### Code Practice:
- [ ] **Create:** `knn_classification.py`
  - Pipeline: StandardScaler + KNN
  - Plot accuracy vs K (elbow method for K)
  - Compare `weights='uniform'` vs `weights='distance'`
  - Decision boundary visualization
  - KNN for regression too: `KNeighborsRegressor`

### Questions (10):
- [ ] Complete all 10 questions

---

## Chapter 7: Naive Bayes

> **Probability-based! Uses Bayes Theorem. "Naive" = assumes features independent**

### Theory:
- [ ] 7.1 **Bayes Theorem**: P(class|features) = P(features|class) × P(class) / P(features)
- [ ] 7.2 **"Naive" Assumption**: Features are independent given class
  - Usually FALSE but works surprisingly well!
- [ ] 7.3 **Types**:
  - GaussianNB: Continuous features (assumes normal distribution)
  - MultinomialNB: Count/frequency data (great for TEXT!)
  - BernoulliNB: Binary features (word present/absent)
- [ ] 7.4 **Spam Detection** (classic Naive Bayes application)
- [ ] 7.5 **Pros**: Super fast, small data friendly, text classification champion
- [ ] 7.6 **Cons**: Independence assumption often wrong, probability calibration issues

### Code Practice:
- [ ] **Create:** `naive_bayes.py`
  - GaussianNB on Iris dataset
  - MultinomialNB on text data (spam detection!)
  - Text vectorization: `CountVectorizer`, `TfidfVectorizer`
  - Compare with Logistic Regression on text

### Questions (10):
- [ ] Complete all 10 questions

---

## Chapter 8: Grand Model Comparison Project

> **All models on ONE dataset — find the BEST!**

### Project:
- [ ] 8.1 Pick a real dataset (Kaggle: Titanic, Heart Disease, or Loan Prediction)
- [ ] 8.2 Full EDA
- [ ] 8.3 Preprocessing pipeline (missing data + encoding + scaling)
- [ ] 8.4 Train ALL 6 models:
  ```
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. SVM
  5. KNN
  6. Naive Bayes
  ```
- [ ] 8.5 Compare with same CV folds (Accuracy, F1, AUC)
- [ ] 8.6 Learning curves for top 3 models
- [ ] 8.7 GridSearchCV for top 2 models
- [ ] 8.8 Final evaluation on test set
- [ ] 8.9 Write report: Which model won and WHY?

---

## Chapter 9: Ensemble Methods — Boosting

> **Trees that LEARN FROM EACH OTHER'S MISTAKES**

### Theory:
- [ ] 9.1 **Bagging vs Boosting**:
  - Bagging (Random Forest): Trees trained independently, then vote
  - Boosting: Trees trained sequentially, each fixing previous errors
- [ ] 9.2 **Gradient Boosting**: Each tree fits the RESIDUAL errors
- [ ] 9.3 **XGBoost**: Fast, regularized gradient boosting (Kaggle king!)
- [ ] 9.4 **Key Concepts**: Learning rate, n_estimators, max_depth
- [ ] 9.5 **AdaBoost**: Earlier boosting method

### Code Practice:
- [ ] **Create:** `boosting.py`
  - GradientBoostingClassifier
  - XGBoost (install: `pip install xgboost`)
  - Compare: RandomForest vs GradientBoosting vs XGBoost
  - Tune with GridSearchCV

### Questions (10):
- [ ] Complete all 10 questions

---

## Summary Checklist

```
PRE-CLASSIFICATION:
  [ ] Bridge: Scaling + Pipelines
  [ ] Bridge: Categorical Encoding

CLASSIFICATION CHAPTERS:
  Ch 1: Logistic Regression        [ ]
  Ch 2: Classification Metrics     [ ]
  Ch 3: Decision Trees             [ ]
  Ch 4: Random Forest              [ ]
  Ch 5: SVM                        [ ]
  Ch 6: KNN                        [ ]
  Ch 7: Naive Bayes                [ ]
  Ch 8: Grand Comparison Project   [ ]
  Ch 9: Ensemble (Boosting)        [ ]
```

---

## 🔮 REMAINING TOPICS (Future — After Classification)

> **These are topics skipped during regression that will be done later**

### Data Preprocessing (do with real project datasets)
- [ ] Missing data handling (SimpleImputer, strategies)
- [ ] Outlier detection and treatment (Z-score, IQR)
- [ ] Power transforms (Box-Cox, Yeo-Johnson)

### PCA & Dimensionality
- [ ] Curse of Dimensionality (why 1000 features = bad)
- [ ] PCA: Geometric intuition + Eigenvalues → Top k components
- [ ] `sklearn.decomposition.PCA`
- [ ] t-SNE for visualization

### Feature Engineering (learn with projects)
- [ ] Creating features (ratios, interactions, time features)
- [ ] Feature selection (RFE, Lasso coefficients, SelectKBest)
- [ ] Feature importance (tree-based, permutation, SHAP)

### Regression Gaps
- [ ] Assumptions deeper practice (diagnostics code)
- [ ] Ridge/Lasso math derivation (L1/L2 penalty origins)
- [ ] P-value with t-test (statistics deep dive)

### After Classification → What's Next?
```
Unsupervised Learning:
  ├── K-Means Clustering
  ├── Hierarchical Clustering
  ├── DBSCAN
  └── Anomaly Detection

Deep Learning:
  ├── Neural Network Basics
  ├── CNN (Images)
  ├── RNN/LSTM (Sequences)
  └── Transformers

Deployment:
  ├── Model Saving/Loading
  ├── Flask/FastAPI
  └── Docker
```
