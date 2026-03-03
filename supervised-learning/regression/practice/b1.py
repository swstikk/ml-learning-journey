from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

# ═══ LOAD DATA ═══
X, y = load_diabetes(return_X_y=True)                         # FIX 1: Direct unpack, no print(data)
print(f"Shape: X={X.shape}, y={y.shape}")                     # Useful info only

# ═══ SPLIT FIRST — ALWAYS! ═══
X_train, X_test, y_train, y_test = train_test_split(          # FIX 2: Split BEFORE fit
    X, y, test_size=0.2, random_state=42
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))               # Create plots upfront

for i, ax in zip(range(1, 4), axes):
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=i, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    train_r2 = r2_score(y_train, pipe.predict(X_train))
    test_r2  = r2_score(y_test, y_pred)
    mae      = mean_absolute_error(y_test, y_pred)

    print(f"\n--- Degree {i} ---")
    print(f"Train R²: {train_r2:.3f}")
    print(f"Test  R²: {test_r2:.3f}")
    print(f"MAE:      {mae:.1f}")

    # ═══ FIX 3: Actual vs Predicted — correct axes! ═══
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)

    # Perfect prediction line (diagonal)
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Perfect')

    ax.set_xlabel('Actual y')
    ax.set_ylabel('Predicted y')
    ax.set_title(f"Degree={i}  |  Test R²={test_r2:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Actual vs Predicted — Poly Degree 1,2,3', fontsize=13)
plt.tight_layout()
plt.savefig('g:/plans/mml/ml_learning_plan/statistics_lessons/code/revision/b1_fixed.png', dpi=100)
plt.show()
print("\nDone! Plot saved as b1_fixed.png")
