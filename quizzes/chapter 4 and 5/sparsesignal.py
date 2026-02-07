from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, LassoCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

import numpy as np 
ols= LinearRegression()
x, y, coef = make_regression(n_samples=100, n_features=100, n_informative=3, noise=10, coef=True, random_state=42)
alfas = [0.1, 1.0, 10.0]
# call aundi :3 alfas
alfasdict={}
r2dict={}
vclasso = LassoCV(alphas=alfas)
vclasso.fit(x,y)
beta=vclasso.coef_

nofcoef=np.sum(abs(beta)>0.01)
foundindices=beta!=0
for i,bbb in zip(coef[foundindices],beta[foundindices]):
    print(f"before = {i}, after = {bbb}")
for alpha in alfas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(x,y)
    print(f"alpha = {alpha}")
    # print(f"coef = {lasso.coef_}")
    # print(f"intercept = {lasso.intercept_}")
    print(f"score = {lasso.score(x,y)}")
    nonzerofeatures=np.sum(abs(lasso.coef_)>0.01)
    print(f"nonzerofeatures = {nonzerofeatures}")
    print("\n")
    alfasdict[alpha]=nonzerofeatures
    r2dict[alpha]=lasso.score(x,y)


# ==0)
print("Creating plot...")


# Plot the results
plt.figure(figsize=(10, 6))
alphas_list = list(alfasdict.keys())
non_zero_list = list(alfasdict.values())

plt.plot(alphas_list, non_zero_list, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Alpha (Regularization Strength)', fontsize=12, fontweight='bold')
plt.ylabel('Number of Non-Zero Features', fontsize=12, fontweight='bold')
plt.title('Lasso Feature Selection: Alpha vs Non-Zero Features', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(alphas_list) 
for alpha, count in zip(alphas_list, non_zero_list):
    plt.text(alpha, count + 2, f'{count}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('sparse_signal_plot.png', dpi=100)
plt.show()

print("\n✅ Plot saved as: sparse_signal_plot.png")
print(f"Best alpha from LassoCV: {vclasso.alpha_}")
print(f"Non-zero features with best alpha: {nofcoef}")
