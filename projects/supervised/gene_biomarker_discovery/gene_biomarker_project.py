"""
=================================================================
 GENE BIOMARKER DISCOVERY PROJECT
 
 Problem:
   150 patients ka RNA sequencing kiya gaya.
   200 genes ka expression measure hua har patient ke liye.
   Target: Disease severity score (higher = worse)
   
   Sirf 15 genes actually disease se related hain.
   Baaki 185 = noise (irrelevant genes).
   
   Task:
   1. Sahi model use karke disease predict karo
   2. Identify karo ki kaun se genes ACTUALLY matter karte hain
   3. Doctor ko report do (less genes = more actionable)
=================================================================
"""

from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ═══════════════════════════════════════
# DATA
# ═══════════════════════════════════════
X, y, true_coef = make_regression(
    n_samples=150,
    n_features=200,
    n_informative=15,
    noise=30,
    coef=True,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

true_causal_genes = np.where(true_coef != 0)[0]
print(f"Dataset ready: {X_train.shape[0]} train, {X_test.shape[0]} test patients")
print(f"Features: {X.shape[1]} genes | Target range: {y.min():.0f} to {y.max():.0f}")
print(f"(There are {len(true_causal_genes)} true causal genes)\n")

# ═══════════════════════════════════════
# TASK 1: Baseline — Plain Linear Regression
# ═══════════════════════════════════════
ols = LinearRegression()
ols.fit(X_train, y_train)
print(f"r2train {ols.score(X_train,y_train):.3f} r2 test {ols.score(X_test,y_test):.3f}")
# r2train 1.0 r2 test ~0.58 → overfit!

# ═══════════════════════════════════════
# TASK 2: Lasso Regression (LassoCV)
# ═══════════════════════════════════════
lasso_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', LassoCV(cv=5))
])
lasso_pipe.fit(X_train, y_train)
coef = lasso_pipe.named_steps['lasso'].coef_

# ═══════════════════════════════════════
# TASK 3: Verify — Did Lasso find the right genes?
# ═══════════════════════════════════════
non_zero = np.where(coef != 0)[0]
correctly_found = np.intersect1d(non_zero, true_causal_genes)
false_positives = np.setdiff1d(non_zero, true_causal_genes)
missed_genes = np.setdiff1d(true_causal_genes, non_zero)

print(f"Correctly identified: {len(correctly_found)} / {len(true_causal_genes)}")
print(f"False Positives: {len(false_positives)}")
print(f"Missed Genes: {len(missed_genes)}")
print(f"r2train {lasso_pipe.score(X_train,y_train):.3f} r2 test {lasso_pipe.score(X_test,y_test):.3f}")

# ═══════════════════════════════════════
# TASK 4: Ridge Comparison
# ═══════════════════════════════════════
alphas = [0.1, 1, 10, 100, 1000]
for i in alphas:
    pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=i))])
    pipe.fit(X_train, y_train)
    print(f"alpha={i} test_r2={pipe.score(X_test,y_test):.3f} nonzero={np.sum(np.abs(pipe.named_steps['ridge'].coef_)>0.01)}")
# Lasso better: Ridge never zeros out genes (not interpretable)

# ═══════════════════════════════════════
# TASK 5: ElasticNet + GridSearchCV
# ═══════════════════════════════════════
paramgd = {
    'elasticnet__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
    'elasticnet__l1_ratio': [0.2, 0.5, 0.7, 0.9, 0.95, 1.0]
}
pipegd = Pipeline([('scaler', StandardScaler()), ('elasticnet', ElasticNet(max_iter=10000))])
gds = GridSearchCV(pipegd, param_grid=paramgd, cv=5, scoring='r2')
gds.fit(X_train, y_train)

enet_test_r2 = r2_score(y_test, gds.best_estimator_.predict(X_test))
lasso_test_r2 = lasso_pipe.score(X_test, y_test)
print(f"Best: {gds.best_params_}")
print(f"ElasticNet Test R2: {enet_test_r2:.3f}  |  Lasso Test R2: {lasso_test_r2:.3f}")
print(f"Winner: {'ElasticNet' if enet_test_r2 > lasso_test_r2 else 'Lasso'}")

# ═══════════════════════════════════════
# TASK 6: Plot — Actual vs Predicted
# ═══════════════════════════════════════
pred = lasso_pipe.predict(X_test)
plt.figure(figsize=(7, 6))
plt.scatter(y_test, pred, alpha=0.7, edgecolors='k', linewidth=0.5, color='steelblue')
lo = min(y_test.min(), pred.min())
hi = max(y_test.max(), pred.max())
plt.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Perfect prediction')
plt.xlabel('Actual Disease Score')
plt.ylabel('Predicted Disease Score')
plt.title(f'Lasso: Actual vs Predicted\nTest R2 = {r2_score(y_test, pred):.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════
# TASK 7: Biomarker Report
# ═══════════════════════════════════════
gene_names = [f"Gene_{i:03d}" for i in range(X.shape[1])]
non_zero_ids = np.where(coef != 0)[0]
gene_importance = [(gene_names[i], abs(coef[i]), i in true_causal_genes) for i in non_zero_ids]
gene_importance_sorted = sorted(gene_importance, key=lambda x: -x[1])

print(f"\n{'Rank':<6} {'Gene':<12} {'Importance':>12}  {'Real Cause?'}")
print("-" * 48)
for rank, (gene, imp, is_real) in enumerate(gene_importance_sorted[:10], 1):
    bar = "=" * int(imp / gene_importance_sorted[0][1] * 15)
    verdict = "YES" if is_real else "False positive"
    print(f"{rank:<6} {gene:<12} {imp:>12.2f}  {verdict}  {bar}")

print("\nDone!")
