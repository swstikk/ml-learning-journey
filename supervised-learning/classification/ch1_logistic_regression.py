"""
═══════════════════════════════════════════════════════════════════════════════
  CLASSIFICATION CH 1: LOGISTIC REGRESSION — Complete Code
═══════════════════════════════════════════════════════════════════════════════

  Topics Covered:
  1. Sigmoid function visualization + derivative
  2. Breast Cancer dataset — LogisticRegression training
  3. Coefficients interpretation (log-odds, odds ratio)
  4. predict() vs predict_proba() comparison
  5. C parameter effect (regularization tuning)
  6. Decision Boundary visualization (2D)
  7. Log Loss from scratch
  8. Logistic Regression from scratch (Gradient Descent!)
  9. Multiclass — Iris dataset
  10. sklearn vs from-scratch comparison

  File: statistics_lessons/code/ch1_logistic_regression.py
═══════════════════════════════════════════════════════════════════════════════
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.use('Agg')



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Plot styling
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

print("=" * 70)
print("  CLASSIFICATION CH 1: LOGISTIC REGRESSION")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: SIGMOID FUNCTION — Visualization + Properties
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SECTION 1: SIGMOID FUNCTION")
print("=" * 70)

def sigmoid(z):
    """Sigmoid function: sigma(z) = 1 / (1 + e^(-z))"""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Derivative: sigma'(z) = sigma(z) * (1 - sigma(z))"""
    s = sigmoid(z)
    return s * (1 - s)

# Verify key values
print("\nSigmoid key values:")
for z_val in [-10, -5, -2, -1, 0, 1, 2, 5, 10]:
    s = sigmoid(z_val)
    print(f"  sigma({z_val:+3d}) = {s:.6f}")

print(f"\nsigma(0) = {sigmoid(0):.4f}  (Exactly 0.5 - midpoint!)")
print(f"sigma(5) = {sigmoid(5):.6f}  (Almost 1)")
print(f"sigma(-5) = {sigmoid(-5):.6f}  (Almost 0)")

# Plot sigmoid + derivative
z = np.linspace(-10, 10, 200)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sigmoid
axes[0].plot(z, sigmoid(z), 'b-', linewidth=2.5, label='$\\sigma(z) = \\frac{1}{1+e^{-z}}$')
axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='y = 0.5')
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.3)
axes[0].scatter([0], [0.5], color='red', s=100, zorder=5)
axes[0].set_xlabel('z (Linear Combination)')
axes[0].set_ylabel('$\\sigma(z)$')
axes[0].set_title('Sigmoid Function', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].set_ylim(-0.05, 1.05)

# Derivative
axes[1].plot(z, sigmoid_derivative(z), 'r-', linewidth=2.5,
             label="$\\sigma'(z) = \\sigma(z)(1-\\sigma(z))$")
axes[1].scatter([0], [0.25], color='red', s=100, zorder=5)
axes[1].annotate('Max = 0.25 at z=0', xy=(0, 0.25), xytext=(3, 0.22),
                 fontsize=11, arrowprops=dict(arrowstyle='->', color='black'))
axes[1].set_xlabel('z')
axes[1].set_ylabel("$\\sigma'(z)$")
axes[1].set_title('Sigmoid Derivative', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig('g:/plans/mml/ml_learning_plan/statistics_lessons/code/ch1_sigmoid_plot.png',
            dpi=120, bbox_inches='tight')
plt.close()
print("\n[SAVED] ch1_sigmoid_plot.png")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: BREAST CANCER DATASET — Binary Classification
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SECTION 2: BREAST CANCER DATASET — First LogisticRegression!")
print("=" * 70)

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names

print(f"\nDataset shape: {X.shape}")
print(f"Features: {len(feature_names)}")
print(f"Classes: {target_names}")
print(f"Class distribution: Malignant(0)={sum(y==0)}, Benign(1)={sum(y==1)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Pipeline: Scaler + LogReg
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(C=1.0, max_iter=10000, random_state=42))
])

pipe.fit(X_train, y_train)

# Results
train_acc = pipe.score(X_train, y_train)
test_acc = pipe.score(X_test, y_test)
print(f"\nTrain Accuracy: {train_acc:.4f}  ({train_acc*100:.1f}%)")
print(f"Test Accuracy:  {test_acc:.4f}  ({test_acc*100:.1f}%)")

# Cross-validation
cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\nClassification Report:")
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_names))


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: COEFFICIENTS INTERPRETATION
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SECTION 3: COEFFICIENTS — Log-Odds aur Odds Ratio")
print("=" * 70)

model = pipe.named_steps['model']
coefs = model.coef_[0]
intercept = model.intercept_[0]

print(f"\nIntercept (w0): {intercept:.4f}")
print(f"\nTop 10 Features by |coefficient| (sabse influential):")
print("-" * 60)

# Sort by absolute value
indices = np.argsort(np.abs(coefs))[::-1]
for i, idx in enumerate(indices[:10]):
    coef = coefs[idx]
    odds_ratio = np.exp(coef)
    print(f"  {i+1}. {feature_names[idx]:30s} | coef = {coef:+.4f} | odds ratio = {odds_ratio:.4f}")

print("\nInterpretation:")
top_idx = indices[0]
top_coef = coefs[top_idx]
top_or = np.exp(top_coef)
print(f"  '{feature_names[top_idx]}' ka coefficient = {top_coef:+.4f}")
if top_coef > 0:
    print(f"  -> 1 std increase se odds of Benign BADHTE hain by {top_or:.2f}x")
else:
    print(f"  -> 1 std increase se odds of Benign GHATTE hain by {1/top_or:.2f}x")

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
top_n = 15
top_indices = indices[:top_n]
colors = ['green' if c > 0 else 'red' for c in coefs[top_indices]]
bars = ax.barh(range(top_n), coefs[top_indices], color=colors, alpha=0.7,
               edgecolor='black', linewidth=0.5)
ax.set_yticks(range(top_n))
ax.set_yticklabels([feature_names[i] for i in top_indices])
ax.set_xlabel('Coefficient Value (Log-Odds)', fontsize=12)
ax.set_title('Logistic Regression — Top 15 Feature Coefficients\n'
             '(Green = increases P(Benign), Red = decreases P(Benign))',
             fontsize=13, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=1)
plt.tight_layout()
plt.savefig('g:/plans/mml/ml_learning_plan/statistics_lessons/code/ch1_coefficients.png',
            dpi=120, bbox_inches='tight')
plt.close()
print("\n[SAVED] ch1_coefficients.png")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: predict() vs predict_proba()
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SECTION 4: predict() vs predict_proba() — The Difference!")
print("=" * 70)

# Get 10 sample predictions
sample_indices = np.random.RandomState(42).choice(len(X_test), 10, replace=False)
X_sample = X_test[sample_indices]
y_sample = y_test[sample_indices]

y_pred_hard = pipe.predict(X_sample)
y_pred_proba = pipe.predict_proba(X_sample)

print(f"\n{'#':>3} | {'Actual':>8} | {'predict()':>10} | {'P(Malig)':>10} | {'P(Benign)':>10} | {'Confidence':>10}")
print("-" * 70)
for i in range(len(X_sample)):
    actual = target_names[y_sample[i]]
    pred = target_names[y_pred_hard[i]]
    p0 = y_pred_proba[i][0]  # P(Malignant)
    p1 = y_pred_proba[i][1]  # P(Benign)
    conf = max(p0, p1)
    marker = "SURE" if conf > 0.95 else "hmm" if conf > 0.7 else "???"
    print(f"  {i+1:>2} | {actual:>8} | {pred:>10} | {p0:>10.4f} | {p1:>10.4f} | {conf:>8.1%} {marker}")

print("\nDekho: predict() sirf 0/1 deta hai, lekin predict_proba() batata hai")
print("ki model KITNA SURE hai. 51% aur 99% dono '1' predict karte hain!")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: C PARAMETER EFFECT — Regularization Tuning
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SECTION 5: C PARAMETER — Regularization ka Effect")
print("=" * 70)

C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_scores = []
test_scores = []

print(f"\n{'C':>10} | {'Train Acc':>10} | {'Test Acc':>10} | {'Regularization':>15}")
print("-" * 55)
for C in C_values:
    pipe_c = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(C=C, max_iter=10000, random_state=42))
    ])
    pipe_c.fit(X_train, y_train)
    tr = pipe_c.score(X_train, y_train)
    te = pipe_c.score(X_test, y_test)
    train_scores.append(tr)
    test_scores.append(te)
    reg_label = "STRONG" if C <= 0.01 else "MODERATE" if C <= 10 else "WEAK"
    print(f"  {C:>8} | {tr:>10.4f} | {te:>10.4f} | {reg_label:>15}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(C_values, train_scores, 'bo-', label='Train Accuracy', linewidth=2, markersize=8)
ax.plot(C_values, test_scores, 'rs-', label='Test Accuracy', linewidth=2, markersize=8)
ax.set_xscale('log')
ax.set_xlabel('C (Regularization Inverse)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('C Parameter Effect on Logistic Regression\n'
             'Small C = Strong Regularization | Large C = Weak Regularization',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='C=1 (default)')
plt.tight_layout()
plt.savefig('g:/plans/mml/ml_learning_plan/statistics_lessons/code/ch1_c_parameter.png',
            dpi=120, bbox_inches='tight')
plt.close()
print("\n[SAVED] ch1_c_parameter.png")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6: DECISION BOUNDARY — 2D Visualization
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SECTION 6: DECISION BOUNDARY — 2D mein Visualize!")
print("=" * 70)

# Use only 2 features for visualization
X_2d = X[:, :2]  # mean radius, mean texture
feature_name_2d = [feature_names[0], feature_names[1]]

X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.2, random_state=42, stratify=y
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, C_val, title in zip(axes, [0.01, 1, 100],
                              ['C=0.01 (Strong Reg)', 'C=1 (Default)', 'C=100 (Weak Reg)']):
    pipe_2d = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(C=C_val, max_iter=10000, random_state=42))
    ])
    pipe_2d.fit(X_train_2d, y_train_2d)

    # Create mesh grid
    scaler = pipe_2d.named_steps['scaler']
    X_scaled = scaler.transform(X_2d)
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict on mesh
    model_2d = pipe_2d.named_steps['model']
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    X_test_scaled = scaler.transform(X_test_2d)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    scatter = ax.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1],
                         c=y_test_2d, cmap='RdYlBu', edgecolors='black',
                         linewidth=0.5, s=50)
    acc = pipe_2d.score(X_test_2d, y_test_2d)
    ax.set_title(f'{title}\nTest Acc: {acc:.3f}', fontsize=12, fontweight='bold')
    ax.set_xlabel(f'{feature_name_2d[0]} (scaled)')
    ax.set_ylabel(f'{feature_name_2d[1]} (scaled)')

plt.suptitle('Decision Boundary — Effect of C on Logistic Regression',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('g:/plans/mml/ml_learning_plan/statistics_lessons/code/ch1_decision_boundary.png',
            dpi=120, bbox_inches='tight')
plt.close()
print("[SAVED] ch1_decision_boundary.png")
print("\nNotice: Decision boundary is ALWAYS a straight line (linear)!")
print("C changes HOW MUCH the model tries to fit training data, not the shape.")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 7: LOG LOSS FROM SCRATCH
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SECTION 7: LOG LOSS — From Scratch Calculation")
print("=" * 70)

def log_loss_manual(y_true, y_prob, eps=1e-15):
    """
    Log Loss = -(1/n) * SUM[ y*log(p) + (1-y)*log(1-p) ]
    eps = small number to avoid log(0)
    """
    y_prob = np.clip(y_prob, eps, 1 - eps)  # Avoid log(0)!
    loss = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return loss

# Test with examples
print("\nLog Loss examples:")
print("-" * 50)

# Perfect predictions
print("\nCase 1: PERFECT predictions (y=1, p=0.99)")
print(f"  Loss = {log_loss_manual(np.array([1]), np.array([0.99])):.6f}  (very low!)")

print("\nCase 2: WRONG predictions (y=1, p=0.01)")
print(f"  Loss = {log_loss_manual(np.array([1]), np.array([0.01])):.6f}  (VERY high!)")

print("\nCase 3: UNCERTAIN predictions (y=1, p=0.50)")
print(f"  Loss = {log_loss_manual(np.array([1]), np.array([0.50])):.6f}  (medium)")

print("\nCase 4: Model ka actual log loss:")
y_proba_test = pipe.predict_proba(X_test)[:, 1]  # P(class 1)
manual_loss = log_loss_manual(y_test, y_proba_test)
print(f"  Manual Log Loss = {manual_loss:.6f}")

from sklearn.metrics import log_loss as sklearn_log_loss
sklearn_loss = sklearn_log_loss(y_test, y_proba_test)
print(f"  sklearn Log Loss = {sklearn_loss:.6f}")
print(f"  Difference = {abs(manual_loss - sklearn_loss):.10f}  (basically zero!)")

# Visualize log loss curves
p_range = np.linspace(0.01, 0.99, 100)
loss_y1 = -np.log(p_range)       # When y=1: -log(p)
loss_y0 = -np.log(1 - p_range)   # When y=0: -log(1-p)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(p_range, loss_y1, 'b-', linewidth=2.5, label='y=1: Loss = -log(p)')
ax.plot(p_range, loss_y0, 'r-', linewidth=2.5, label='y=0: Loss = -log(1-p)')
ax.set_xlabel('Predicted Probability (p)', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Log Loss: Penalty Increases SHARPLY for Confident Wrong Predictions!',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=12)
ax.annotate('y=1, p=0.01\nLoss = 4.6!', xy=(0.01, 4.6), xytext=(0.15, 4.0),
            fontsize=11, arrowprops=dict(arrowstyle='->', color='blue'))
ax.annotate('y=0, p=0.99\nLoss = 4.6!', xy=(0.99, 4.6), xytext=(0.75, 4.0),
            fontsize=11, arrowprops=dict(arrowstyle='->', color='red'))
plt.tight_layout()
plt.savefig('g:/plans/mml/ml_learning_plan/statistics_lessons/code/ch1_log_loss.png',
            dpi=120, bbox_inches='tight')
plt.close()
print("\n[SAVED] ch1_log_loss.png")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 8: LOGISTIC REGRESSION FROM SCRATCH!
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SECTION 8: LOGISTIC REGRESSION FROM SCRATCH (Gradient Descent)")
print("=" * 70)

class LogisticRegressionScratch:
    """
    Logistic Regression implemented from scratch using Gradient Descent.

    Math:
      z = X @ w + b
      p = sigmoid(z) = 1/(1+e^(-z))
      Loss = -(1/n) * sum[ y*log(p) + (1-y)*log(1-p) ]

      Gradients:
      dw = (1/n) * X.T @ (p - y)
      db = (1/n) * sum(p - y)
    """

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.n_iters):
            # Forward pass
            z = X @ self.weights + self.bias
            p = sigmoid(z)

            # Compute loss
            eps = 1e-15
            p_clipped = np.clip(p, eps, 1 - eps)
            loss = -np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
            self.loss_history.append(loss)

            # Compute gradients
            error = p - y  # (n_samples,)
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if (i + 1) % 200 == 0:
                print(f"  Iteration {i+1:>5d} | Loss: {loss:.6f}")

        return self

    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# Train from scratch
print("\nTraining from scratch...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

my_model = LogisticRegressionScratch(learning_rate=0.1, n_iters=1000)
my_model.fit(X_train_scaled, y_train)

scratch_train_acc = my_model.score(X_train_scaled, y_train)
scratch_test_acc = my_model.score(X_test_scaled, y_test)

print(f"\nFrom Scratch Results:")
print(f"  Train Accuracy: {scratch_train_acc:.4f}")
print(f"  Test Accuracy:  {scratch_test_acc:.4f}")

print(f"\nsklearn Results (for comparison):")
print(f"  Train Accuracy: {train_acc:.4f}")
print(f"  Test Accuracy:  {test_acc:.4f}")

print(f"\nDifference: {abs(test_acc - scratch_test_acc):.4f}")
print("(sklearn uses optimized solvers like L-BFGS, itna difference expected hai!)")

# Plot loss curve
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(my_model.loss_history, 'b-', linewidth=1.5)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Log Loss', fontsize=12)
ax.set_title('From-Scratch Logistic Regression: Loss Convergence\n'
             'Dekho kaise smoothly converge ho raha hai (CONVEX loss!)',
             fontsize=13, fontweight='bold')
ax.axhline(y=my_model.loss_history[-1], color='red', linestyle='--',
           alpha=0.5, label=f'Final Loss: {my_model.loss_history[-1]:.4f}')
ax.legend()
plt.tight_layout()
plt.savefig('g:/plans/mml/ml_learning_plan/statistics_lessons/code/ch1_loss_curve.png',
            dpi=120, bbox_inches='tight')
plt.close()
print("\n[SAVED] ch1_loss_curve.png")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 9: MULTICLASS — Iris Dataset
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SECTION 9: MULTICLASS — Iris Dataset (3 Classes)")
print("=" * 70)

iris = load_iris()
X_iris, y_iris = iris.data, iris.target
print(f"\nIris Dataset: {X_iris.shape}")
print(f"Classes: {iris.target_names}")
print(f"Distribution: {np.bincount(y_iris)}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
)

# OvR vs Multinomial comparison
print("\n--- OvR (One-vs-Rest) ---")
ovr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(multi_class='ovr', max_iter=10000, random_state=42))
])
ovr_model.fit(X_tr, y_tr)
ovr_acc = ovr_model.score(X_te, y_te)
print(f"  OvR Accuracy: {ovr_acc:.4f}")

print("\n--- Multinomial (Softmax) ---")
multi_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(multi_class='multinomial', solver='lbfgs',
                                  max_iter=10000, random_state=42))
])
multi_model.fit(X_tr, y_tr)
multi_acc = multi_model.score(X_te, y_te)
print(f"  Multinomial Accuracy: {multi_acc:.4f}")

# Show predict_proba for multiclass
sample = X_te[:5]
proba_ovr = ovr_model.predict_proba(sample)
proba_multi = multi_model.predict_proba(sample)

print(f"\nProbabilities for first 5 test samples:")
print(f"{'':>5} {'Setosa':>10} {'Versicolor':>12} {'Virginica':>12} | {'Predicted':>10}")
print("-" * 55)
for i in range(5):
    p = proba_multi[i]
    pred = iris.target_names[np.argmax(p)]
    print(f"  {i+1:>2} | {p[0]:>8.4f}   {p[1]:>10.4f}   {p[2]:>10.4f} | {pred:>10}")

print("\nDekho: Har row ke probabilities ka sum = 1.0 (Softmax guarantee!)")
for i in range(3):
    print(f"  Row {i+1} sum = {proba_multi[i].sum():.4f}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 10: WHY LINEAR REGRESSION FAILS — Visual Demo
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SECTION 10: WHY LINEAR REGRESSION FAILS for Classification")
print("=" * 70)

from sklearn.linear_model import LinearRegression

# Simple 1D example
np.random.seed(42)
X_demo = np.sort(np.random.uniform(0, 10, 30)).reshape(-1, 1)
y_demo = (X_demo.ravel() > 5).astype(int)

# Add outlier to show problem
X_demo_outlier = np.vstack([X_demo, [[20]]])
y_demo_outlier = np.append(y_demo, 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, X_d, y_d, title in zip(
    axes,
    [X_demo, X_demo_outlier],
    [y_demo, y_demo_outlier],
    ['Without Outlier', 'With Outlier (X=20)']
):
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_d, y_d)
    X_line = np.linspace(-2, 22, 200).reshape(-1, 1)
    y_lin = lr.predict(X_line)

    # Logistic Regression
    log_pipe = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=10000))])
    log_pipe.fit(X_d, y_d)
    y_log = log_pipe.predict_proba(X_line)[:, 1]

    ax.scatter(X_d, y_d, c=y_d, cmap='RdYlBu', edgecolors='black', s=80, zorder=5)
    ax.plot(X_line, y_lin, 'r--', linewidth=2, label='Linear Regression', alpha=0.8)
    ax.plot(X_line, y_log, 'b-', linewidth=2.5, label='Logistic Regression')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axhline(y=1, color='gray', linewidth=0.5)
    ax.axhline(y=0.5, color='green', linewidth=1, linestyle=':', alpha=0.5, label='Threshold 0.5')
    ax.fill_between(X_line.ravel(), -0.3, 0, alpha=0.1, color='red', label='<0 (Invalid!)')
    ax.fill_between(X_line.ravel(), 1, 1.3, alpha=0.1, color='red', label='>1 (Invalid!)')
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel('Feature X', fontsize=12)
    ax.set_ylabel('Predicted Value / Probability', fontsize=12)
    ax.set_title(f'{title}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='center left')

plt.suptitle('Linear vs Logistic Regression for Classification\n'
             'Linear Regression predicts values OUTSIDE [0,1] range!',
             fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('g:/plans/mml/ml_learning_plan/statistics_lessons/code/ch1_linear_vs_logistic.png',
            dpi=120, bbox_inches='tight')
plt.close()
print("[SAVED] ch1_linear_vs_logistic.png")
print("\nLinear Regression predictions go below 0 and above 1 = INVALID probabilities!")
print("Logistic Regression ALWAYS stays in [0, 1] range. That's the power of sigmoid!")


# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  CHAPTER 1 COMPLETE: LOGISTIC REGRESSION SUMMARY")
print("=" * 70)

print("""
  WHAT WE COVERED:
  1. Sigmoid function: sigma(z) = 1/(1+e^(-z))
     - Maps (-inf, +inf) to [0, 1]
     - Derivative: sigma(z) * (1 - sigma(z))

  2. Breast Cancer classification:
     - Pipeline: StandardScaler + LogisticRegression
     - Test Accuracy: {:.1f}%

  3. Coefficients = Log-Odds
     - Positive coef => increases P(class 1)
     - Odds Ratio = e^(coefficient)

  4. predict() vs predict_proba()
     - predict() = hard labels (0 or 1)
     - predict_proba() = confidences [P(0), P(1)]

  5. C parameter = 1/lambda (INVERSE of regularization)
     - Small C = strong regularization = simpler model
     - Large C = weak regularization = complex model

  6. Decision Boundary = ALWAYS LINEAR (for basic LogReg)

  7. Log Loss = -(1/n) * sum[y*log(p) + (1-y)*log(1-p)]
     - Derived from Maximum Likelihood Estimation
     - CONVEX => gradient descent guaranteed converges

  8. From-Scratch implementation with Gradient Descent
     - Scratch accuracy: {:.1f}%, sklearn: {:.1f}%

  9. Multiclass: OvR vs Multinomial (Softmax)

  10. Why Linear Regression fails for classification

  KEY INSIGHT: A single neuron in Neural Network = Logistic Regression!
  This is your building block for Deep Learning!

  FILES CREATED:
  - ch1_sigmoid_plot.png
  - ch1_coefficients.png
  - ch1_c_parameter.png
  - ch1_decision_boundary.png
  - ch1_log_loss.png
  - ch1_loss_curve.png
  - ch1_linear_vs_logistic.png
""".format(test_acc * 100, scratch_test_acc * 100, test_acc * 100))

print("NEXT: Chapter 2 — Classification Metrics (Confusion Matrix,")
print("       Precision, Recall, F1, ROC-AUC)")
print("=" * 70)
