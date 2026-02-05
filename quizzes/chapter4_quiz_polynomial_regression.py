"""
================================================================================
CHAPTER 4 QUIZ: Polynomial Regression
================================================================================
Author: Swastik (@swstikk)
Score: 32/35

This file contains my practice solutions for the Polynomial Regression quiz.
Each question is clearly marked with the question statement and my solution.
================================================================================
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ================================================================================
# Q1: Design Matrix Construction (5 marks)
# ================================================================================
"""
QUESTION:
Given data:
    x = [1, 2, 3]
    y = [1, 8, 27]  (y = x³)

Tasks:
1. Write the Design Matrix X for degree=3 polynomial (with intercept column)
2. What will be the shape of X?
3. If beta = (X^T @ X)^(-1) @ X^T @ Y, what shape will beta have?
"""

print("=" * 60)
print("Q1: Design Matrix Construction")
print("=" * 60)

x = pd.DataFrame([1, 2, 3])
Y = pd.DataFrame([1, 8, 27])  # y = x³

poly1 = PolynomialFeatures(degree=3, include_bias=True)
X = poly1.fit_transform(x)

print("Design Matrix X:")
print(X)
print(f"\nX.shape = {X.shape}")  # Expected: (3, 4)

beta = np.linalg.inv(X.T @ X) @ X.T @ Y
print(f"beta.shape = {beta.shape}")  # Expected: (4, 1)

"""
ANSWER:
X = [[1, 1, 1, 1],    # [1, x, x², x³]
     [1, 2, 4, 8],
     [1, 3, 9, 27]]
     
Shape of X: (3, 4) ✅
Shape of beta: (4, 1) ✅
"""

# ================================================================================
# Q2: Overfitting Analysis (6 marks)
# ================================================================================
"""
QUESTION:
Given this table:
| Degree | Train R² | Test R² |
|--------|----------|---------|
| 1      | 0.60     | 0.58    |
| 2      | 0.85     | 0.83    |
| 3      | 0.92     | 0.89    |
| 5      | 0.97     | 0.75    |
| 10     | 0.995    | 0.40    |

Questions:
1. Which degree is overfitting? How do you know?
2. Which degree would you choose for production? Why?
3. What's the gap between Train R² and Test R² called?
4. How would you prevent overfitting?
"""

print("\n" + "=" * 60)
print("Q2: Overfitting Analysis")
print("=" * 60)

"""
MY ANSWERS:
1. Degree 10 is overfitting because test R² (0.40) is MUCH lower than 
   train R² (0.995). The model memorized training data but fails on new data.

2. Degree 3 is best because:
   - Train R² = 0.92 (good fit)
   - Test R² = 0.89 (generalizes well)
   - Small gap between train and test

3. The gap is called "Generalization Gap" or "Variance"

4. Prevent overfitting by:
   - Using lower polynomial degree
   - Getting more training data
   - Using Regularization (Ridge/Lasso)
   - Cross-validation to find optimal degree
"""

# ================================================================================
# Q3: Code Debugging (5 marks)
# ================================================================================
"""
QUESTION:
Find 3 errors in this code and write the correct version.

ORIGINAL (BUGGY) CODE:
X = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y)
model = LinearRegression()
model.fit(X_train, y_train)
X_test_poly = poly.fit_transform(X_test)  # BUG!
y_pred = model.predict(X_test_poly)
"""

print("\n" + "=" * 60)
print("Q3: Code Debugging - CORRECTED CODE")
print("=" * 60)

# CORRECTED VERSION:
# Bug 1: X needs reshape(-1, 1)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25]).reshape(-1, 1)

# Bug 2: Split BEFORE polynomial transformation!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_train)

model = LinearRegression()
model.fit(X_poly, y_train)

# Bug 3: Use transform(), NOT fit_transform() on test data!
y_pred = model.predict(poly.transform(X_test))

print(f"Intercept (b): {model.intercept_}")  # ~0 (computer precision)
print(f"Coefficients (m): {model.coef_}")    # [0, 1] meaning y = x²
print(f"Predictions: {y_pred.flatten()}")

"""
BUGS IDENTIFIED:
1. X was not reshaped to 2D array
2. Split was done AFTER poly transformation (should be BEFORE)
3. Used fit_transform on test data (should use transform only)
"""

# ================================================================================
# Q4: Mathematical Understanding (6 marks)
# ================================================================================
"""
QUESTION:
Given: y = 5 + 2x + 3x²

Questions:
1. Is this model LINEAR or NON-LINEAR? Explain.
2. If we use matrix OLS: beta = (X^T @ X)^(-1) @ X^T @ Y, what will beta be?
3. Why can we use LINEAR regression to fit this CURVE?
"""

print("\n" + "=" * 60)
print("Q4: Mathematical Understanding")
print("=" * 60)

"""
MY ANSWERS:

1. This is a LINEAR model in terms of coefficients!
   - We are finding β₀, β₁, β₂ (linear combination)
   - The CURVE is in x, but the UNKNOWNS are linear
   - y = β₀ + β₁*x + β₂*x²

2. beta = [5, 2, 3]
   - β₀ = 5 (intercept)
   - β₁ = 2 (coefficient of x)
   - β₂ = 3 (coefficient of x²)

3. We treat x² as a NEW FEATURE!
   - Original: 1 feature (x)
   - After PolynomialFeatures: 2 features (x, x²)
   - LinearRegression on 2 features = finding best hyperplane
   - The curve appears because we plot against original x
"""

# ================================================================================
# Q5: Practical Scenario (8 marks)
# ================================================================================
"""
QUESTION:
Temperature vs Ice Cream Sales data (curved relationship):

| Temp (°C) | Sales |
|-----------|-------|
| 10        | 100   |
| 20        | 400   |
| 30        | 700   |
| 40        | 600   |

Write complete code to:
1. Create polynomial features (degree 2)
2. Split data (80% train, 20% test)
3. Train LinearRegression
4. Print coefficients
5. Calculate R² on test set
"""

print("\n" + "=" * 60)
print("Q5: Practical Scenario - Temperature vs Sales")
print("=" * 60)

# Data
temp = np.array([10, 20, 30, 40]).reshape(-1, 1)
sales = np.array([100, 400, 700, 600]).reshape(-1, 1)

# Split FIRST
X_train, X_test, y_train, y_test = train_test_split(
    temp, sales, test_size=0.2, random_state=42
)

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Results
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")
r2_test = r2_score(y_test, model.predict(X_test_poly))
print(f"Test R²: {r2_test}")

# ================================================================================
# Q6: Transform vs Fit_Transform (5 marks)
# ================================================================================
"""
QUESTION:
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)    # Line A
# vs
X_test_poly = poly.fit_transform(X_test)  # Line B

Questions:
1. What's the difference between Line A and Line B?
2. Which one is CORRECT for test data?
3. What happens if you use the WRONG method?
4. Why does PolynomialFeatures need to "fit" at all?
"""

print("\n" + "=" * 60)
print("Q6: Transform vs Fit_Transform")
print("=" * 60)

"""
MY ANSWERS:

1. DIFFERENCE:
   - fit_transform(): LEARNS structure from data + APPLIES transformation
   - transform(): ONLY applies (uses what was already learned)

2. Line A is CORRECT for test data.
   - We should NOT fit on test data, only transform it.

3. If WRONG method (fit_transform on test):
   - Data leakage occurs
   - Model "sees" test data patterns before testing
   - Artificially inflated test scores
   - Model fails on truly new data in production

4. Why fit() is needed:
   - To know: how many input features exist
   - To know: what degree of polynomial to create
   - To store: which columns to generate (x, x², x³, etc.)
   - Without fit(), poly doesn't know what transformations to apply!
   
   IMPORTANT: fit() stores internal state. Calling fit() on test data 
   OVERWRITES the state learned from training data!
"""

print("\n" + "=" * 60)
print("QUIZ COMPLETED! Score: 32/35")
print("=" * 60)
