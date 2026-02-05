"""
================================================================================
CHAPTER 1-3 QUIZ: Simple & Multiple Linear Regression + Diagnostics
================================================================================
Author: Swastik (@swstikk)

This file contains my practice solutions for:
- OLS Formula implementation
- R² calculation (manual and sklearn)
- Coefficient interpretation
- Residual plot analysis
================================================================================
"""

import numpy as np 
import pandas as pd 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ================================================================================
# LOAD DATA: California Housing Dataset
# ================================================================================

print("=" * 60)
print("Loading California Housing Dataset")
print("=" * 60)

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Price"] = data.target
df = df.sample(n=500, random_state=42)

print(f"Dataset shape: {df.shape}")
print(f"Features: {list(df.columns[:-1])}")

# ================================================================================
# Q1: OLS Formula Manual Calculation
# ================================================================================
"""
QUESTION:
Calculate slope (m) and intercept (b) manually using:
- m = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
- b = ȳ - m·x̄
"""

print("\n" + "=" * 60)
print("Q1: Manual OLS Calculation (Single Feature)")
print("=" * 60)

# Using single feature for simple OLS
x = df['MedInc'].values
y = df['Price'].values

# Manual calculation
x_mean = np.mean(x)
y_mean = np.mean(y)

numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)

m_manual = numerator / denominator
b_manual = y_mean - m_manual * x_mean

print(f"Manual calculation:")
print(f"  Slope (m) = {m_manual:.4f}")
print(f"  Intercept (b) = {b_manual:.4f}")

# Verify with sklearn
model_simple = LinearRegression()
model_simple.fit(x.reshape(-1, 1), y)
print(f"\nSklearn verification:")
print(f"  Slope (m) = {model_simple.coef_[0]:.4f}")
print(f"  Intercept (b) = {model_simple.intercept_:.4f}")

# ================================================================================
# Q2: Multiple Linear Regression
# ================================================================================

print("\n" + "=" * 60)
print("Q2: Multiple Linear Regression")
print("=" * 60)

X = df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms']]
Y = df['Price']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    sign = "+" if coef > 0 else "-"
    print(f"  {feature}: {coef:.4f} ({sign} means price {'increases' if coef > 0 else 'decreases'})")
print(f"  Intercept: {model.intercept_:.4f}")

# ================================================================================
# Q3: R² Calculation (Manual vs Sklearn)
# ================================================================================

print("\n" + "=" * 60)
print("Q3: R² Calculation")
print("=" * 60)

# Manual R² calculation
SS_res = np.sum((Y_test - predictions) ** 2)  # Sum of Squared Residuals
SS_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)  # Total Sum of Squares
r2_manual = 1 - (SS_res / SS_tot)

# Sklearn R²
r2_sklearn = model.score(X_test, Y_test)
r2_metric = r2_score(Y_test, predictions)

print(f"R² Manual:  {r2_manual:.4f}")
print(f"R² Sklearn: {r2_sklearn:.4f}")
print(f"R² Metric:  {r2_metric:.4f}")

# ================================================================================
# Q4: RMSE Calculation
# ================================================================================

print("\n" + "=" * 60)
print("Q4: RMSE Calculation")
print("=" * 60)

# Manual RMSE
rmse_manual = np.sqrt(np.mean((Y_test - predictions) ** 2))

# Sklearn RMSE
rmse_sklearn = np.sqrt(mean_squared_error(Y_test, predictions))

print(f"RMSE Manual:  {rmse_manual:.4f}")
print(f"RMSE Sklearn: {rmse_sklearn:.4f}")

# ================================================================================
# Q5: Feature Impact Analysis
# ================================================================================

print("\n" + "=" * 60)
print("Q5: Which Feature Has Most Impact?")
print("=" * 60)

# Absolute values of coefficients
impact = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_,
    'Abs_Impact': np.abs(model.coef_)
}).sort_values('Abs_Impact', ascending=False)

print(impact.to_string(index=False))
print(f"\nMost impactful feature: {impact.iloc[0]['Feature']}")

# ================================================================================
# Q6: Residual Plot (Diagnostics)
# ================================================================================

print("\n" + "=" * 60)
print("Q6: Residual Plot")
print("=" * 60)

residuals = Y_test - predictions

plt.figure(figsize=(10, 5))

# Residuals vs Fitted
plt.subplot(1, 2, 1)
plt.scatter(predictions, residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Fitted Values (Predictions)')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted\n(Should be random cloud)')

# Actual vs Predicted
plt.subplot(1, 2, 2)
plt.scatter(Y_test, predictions, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted\n(Should be on diagonal)')

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=100)
plt.show()

print("Residual plot saved as: residual_analysis.png")

print("\n" + "=" * 60)
print("QUIZ COMPLETED!")
print("=" * 60)
