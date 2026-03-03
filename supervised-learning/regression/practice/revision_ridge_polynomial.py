
import matplotlib.pyplot as plt
import numpy as np                                             # BUG FIX 1: np missing tha!
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
print(f"Shape: X={X.shape}, y={y.shape}")

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

r2r = {}
for i in alphas:
    # BUG FIX 2: Pipeline syntax galat tha!
    # ❌ TU NE LIKHA: Pipeline([( 'poly', X, 'scaler', X, 'ridge', X )])
    #    Ek badi tuple — Pipeline ko 2-element tuples ki LIST chahiye!
    # ✅ SAHI: Pipeline([ ('poly', X), ('scaler', X), ('ridge', X) ])
    pipe = Pipeline([
        ('poly',   PolynomialFeatures(degree=3, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge',  Ridge(alpha=i))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2r[i] = {
        "tr2":      pipe.score(X_train, y_train),
        "te2":      r2_score(y_test, y_pred),
        "mae":      mean_absolute_error(y_test, y_pred),
        "zero_coef": np.sum(np.abs(pipe.named_steps['ridge'].coef_) < 1e-10)
    }
    print(f"alpha={i:6} | Train R2={r2r[i]['tr2']:.3f} | Test R2={r2r[i]['te2']:.3f} | MAE={r2r[i]['mae']:.1f}")

# ══════════════════════════════════════════════════════════════
# LOG SCALE PLOTTING — Kaise aur Kyun
# ══════════════════════════════════════════════════════════════
# Alphas: [0.001, 0.01, 0.1, 1, 10, 100, 1000]
#
# Normal X-axis mein dikhta:
# 0.001..............................................1000  ← pehle 3 bilkul saath!
#
# Log X-axis mein dikhta:
# 0.001 . 0.01 . 0.1 . 1 . 10 . 100 . 1000       ← equal spacing! 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ────────────────────────────────────────────
# LEFT PLOT: Normal scale (bad for this data)
# ────────────────────────────────────────────
ax1.plot(alphas, [r2r[a]['tr2'] for a in alphas],
         'b-o', label='Train R2', linewidth=2)
ax1.plot(alphas, [r2r[a]['te2'] for a in alphas],
         'r-o', label='Test R2', linewidth=2)
ax1.set_xlabel('Alpha (normal scale)')
ax1.set_ylabel('R2 Score')
ax1.set_title('Normal Scale — Pehle 3 points ek jagah dikhe!')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linewidth=0.5)

# ────────────────────────────────────────────
# RIGHT PLOT: Log scale (correct way)
# plt.semilogx() = X-axis log, Y-axis normal
# plt.semilogy() = X-axis normal, Y-axis log
# plt.loglog()   = dono log
# ────────────────────────────────────────────
ax2.semilogx(alphas, [r2r[a]['tr2'] for a in alphas],
             'b-o', label='Train R2', linewidth=2, markersize=8)
ax2.semilogx(alphas, [r2r[a]['te2'] for a in alphas],
             'r-o', label='Test R2', linewidth=2, markersize=8)

# Mark best alpha
best_alpha = max(r2r, key=lambda a: r2r[a]['te2'])
ax2.axvline(x=best_alpha, color='green', linestyle='--', alpha=0.8,
            label=f'Best alpha={best_alpha}')

# Shade zones
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_xlabel('Alpha (LOG scale) ← SAHI TARIKA')
ax2.set_ylabel('R2 Score')
ax2.set_title('Log Scale — Sab points equally spaced!')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')   # which='both' = major + minor gridlines

# Add zone annotations
ax2.text(0.001, -2, 'HIGH VARIANCE\n(overfit)', fontsize=9, color='red', ha='left')
ax2.text(500,   -2, 'HIGH BIAS\n(underfit)', fontsize=9, color='blue', ha='center')

plt.suptitle('Log Scale vs Normal Scale for Alpha Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('g:/plans/mml/ml_learning_plan/statistics_lessons/code/revision/b2_fixed.png', dpi=100)
plt.show()

print(f"\nBest alpha: {best_alpha}")
print(f"Best Test R2: {r2r[best_alpha]['te2']:.3f}")
print("\nPlot saved: b2_fixed.png")
