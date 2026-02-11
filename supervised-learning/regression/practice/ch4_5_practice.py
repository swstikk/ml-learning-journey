import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(42)
X = np.sort(np.random.rand(20, 1) * 10, axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.5, 20)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
deg = [1, 3, 10, 20]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_plot = np.linspace(0, 10, 100).reshape(-1, 1)

for idx, i in enumerate(deg):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    poly = PolynomialFeatures(degree=i, include_bias=False)
    poly_x = poly.fit_transform(x_train)
    poly_x_test = poly.transform(x_test)
    X_plot_poly = poly.transform(X_plot)
    
    ols = LinearRegression()
    ols.fit(poly_x, y_train)
    
    train_r2 = ols.score(poly_x, y_train)
    test_r2 = ols.score(poly_x_test, y_test)
    train_mse = mean_squared_error(y_train, ols.predict(poly_x))
    test_mse = mean_squared_error(y_test, ols.predict(poly_x_test))
    
    ax.scatter(x_train, y_train, color='blue', s=50, alpha=0.6, label='Train data')
    ax.scatter(x_test, y_test, color='green', s=50, alpha=0.6, label='Test data')
    
    y_plot = ols.predict(X_plot_poly)
    ax.plot(X_plot, y_plot, color='red', linewidth=2, label=f'Degree {i} fit')
    
    ax.text(0.05, 0.95, f'Train R²: {train_r2:.3f}\nTest R²: {test_r2:.3f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title(f'Degree {i}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 2)
    
    print(f"\nDegree {i}:")
    print(f"  Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    print(f"  Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}")
    if test_r2 < 0:
        print(f"OVERFITTING! Test! R²! is negative!")

plt.tight_layout()
plt.savefig('polynomial_overfitting.png', dpi=120)
plt.show()
