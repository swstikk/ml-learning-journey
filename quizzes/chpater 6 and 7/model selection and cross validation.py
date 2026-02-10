"""
California Housing dataset par:
1. Ridge ke liye learning curve plot karo
2. Use sklearn.model_selection.learning_curve
3. Train sizes: 10% se 100%
4. Plot train score AND validation score vs. training size
5. Diagnosis likho: Overfit? Underfit? Good fit?
"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import learning_curve, validation_curve, GridSearchCV, KFold, cross_val_score, TimeSeriesSplit, RandomizedSearchCV, train_test_split
import pandas as pd
import time
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt

house =fetch_california_housing()
x=house.data
y=house.target
 

trainsize, trainscore, valscore = learning_curve(
    estimator=Ridge(), X=x, y=y, 
    train_sizes=np.linspace(0.1, 1.0, 10), 
    cv=5, scoring="r2"
)
print(trainsize)
print(trainscore.mean(axis=1))
print(valscore.mean(axis=1))
plt.plot(trainsize,trainscore.mean(axis=1), label="train")
plt.plot(trainsize,valscore.mean(axis=1),label="r2")
plt.legend()
plt.grid(True)
plt.show()

"""
California Housing dataset par:
1. Ridge model ke alpha parameter ke liye validation curve plot karo
2. Alpha range: [0.001, 0.01, 0.1, 1, 10, 100, 1000]
3. Use sklearn.model_selection.validation_curve
4. Plot train score AND validation score vs. alpha
5. Best alpha identify karo from the plot
6. Verify with GridSearchCV ki same answer aata hai
"""

param_grid={"alpha":[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
tc,vs=validation_curve(estimator=Ridge(),X=x,y=y, param_name='alpha',param_range=[0.001, 0.01, 0.1, 1, 10, 100, 1000],cv=5,scoring='r2')


print(tc.mean(axis=1))
print(vs.mean(axis=1))
plt.plot(param_grid["alpha"],tc.mean(axis=1),label="train")
plt.plot(param_grid["alpha"],vs.mean(axis=1),label="r2")
plt.legend()
plt.grid(True)
plt.show()# plot karne par mujeh 100 par peak dikha vs (validation score kaa ) and train score kam hota dikha apne peak se
grid=GridSearchCV(Ridge(),param_grid=param_grid,cv=5,scoring='r2')
grid.fit(x,y)
print("best parma",grid.best_params_)# 100 = upar wala 100 hurrayyy 
print(grid.best_score_)
# print(grid.)


"""
Synthetic data generate karo with non-linear relationship.
Three models train karo:
  a) LinearRegression (potential underfit)
  b) PolynomialFeatures(degree=3) + LinearRegression (potential good fit)
  c) PolynomialFeatures(degree=15) + LinearRegression (potential overfit)

Har model ke liye:
  1. Learning curve plot karo
  2. Identify: Overfit? Underfit? Good?
  3. Sabko ek figure mein 3 subplots mein dikhao
"""




# 1. Synthetic data generation (Non-linear relationship: y = x^3 - 3x + noise)
np.random.seed(42)
X_syn = np.linspace(-3, 3, 200).reshape(-1, 1)
y_syn = (X_syn**3 - 3*X_syn + np.random.normal(0, 3, X_syn.shape)).ravel()

degrees = [1, 3, 15]
titles = ["Degree 1: Underfit (High Bias)", "Degree 3: Good Fit", "Degree 15: Overfit (High Variance)"]

plt.figure(figsize=(18, 5))

for i, degree in enumerate(degrees):
    # model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model=Pipeline([("poly",PolynomialFeatures(degree)),("linear",LinearRegression())])
 
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_syn, y_syn, cv=5, scoring='neg_mean_squared_error', 
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    
    # Convert negative MSE to positive error for plotting
    train_error = -np.mean(train_scores, axis=1)
    test_error = -np.mean(test_scores, axis=1)

    plt.subplot(1, 3, i + 1)
    plt.plot(train_sizes, train_error, 'o-', color="r", label="Train Error")
    plt.plot(train_sizes, test_error, 'o-', color="g", label="Val Error")
    plt.title(titles[i])
    plt.xlabel("Training Examples")
    plt.ylabel("MSE")
    plt.legend(loc="best")
    plt.grid(True)
    if degree == 15: plt.ylim(0, 50) # Limit y-axis for better visibility of the gap

plt.tight_layout()
plt.show()






"""
California Housing data:
Compare these 5 models using SAME KFold splits:
  1. LinearRegression
  2. Ridge(alpha=1)
  3. Lasso(alpha=0.1)
  4. ElasticNet(alpha=0.1, l1_ratio=0.5)
  5. Pipeline: StandardScaler + Ridge(alpha=10)

Requirements:
  a) Use KFold(n_splits=5, shuffle=True, random_state=42)
  b) Print mean ± std for each model
  c) Bar chart banao comparing all models
  d) Best model select karo with justification
  e) Occam's Razor apply karo: kya simplest model kaam kar raha hai?
"""


kfold = KFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "linear": LinearRegression(),
    "ridge": Ridge(alpha=1),
    "lasso": Lasso(alpha=0.1),
    "elasticnet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "pipeline": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10))])
}

all_stds = []
model_names = []

plt.figure(figsize=(15, 8))

for i, (name, model) in enumerate(models.items()):
    scores = cross_val_score(estimator=model, X=x, y=y, cv=kfold, scoring='r2')
    
    # Store for 6th plot
    all_stds.append(scores.std())
    model_names.append(name)
    
    # Plot 1 to 5 (i ranges from 0 to 4, so i+1 is 1 to 5)
    plt.subplot(2, 3, i + 1) 
    plt.bar(range(1, 6), scores, color='skyblue')
    plt.xlabel("Folds")
    plt.ylabel("R² Score")
    plt.title(f"{name}\nMean: {scores.mean():.4f}")
    plt.ylim(0, 1)

# 6th plot: Comparison of Stability (STD)
plt.subplot(2, 3, 6)
plt.bar(model_names, all_stds, color='salmon')
plt.title("Model Stability (Lower Std is Better)")
plt.ylabel("Standard Deviation")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

"""
Trading model scenario:
  1. Generate 3 years of fake stock data (np.random.seed(42))
  2. Create features: lag_1, lag_5, lag_30, rolling_mean_7, rolling_std_7
  3. Build Pipeline: StandardScaler → Ridge
  4. Use GridSearchCV with TimeSeriesSplit(n_splits=5)
  5. Test alphas: [0.01, 0.1, 1, 10, 100]
  6. Print:
     - best alpha
     - best CV score
     - all fold scores
  7. Plot: CV scores for each alpha (bar chart)
"""
# Generate 3 years of fake stock data
np.random.seed(42)
days = 365 * 3
price = [100]
for i in range(1, days):
    price.append(price[-1] * (1 + np.random.normal(0.0003, 0.02)))
df_trade = pd.DataFrame({'price': price})

# Create features: lag_1, lag_5, lag_30, rolling_mean_7, rolling_std_7
df_trade['lag_1'] = df_trade['price'].shift(1)
df_trade['lag_5'] = df_trade['price'].shift(5)
df_trade['lag_30'] = df_trade['price'].shift(30)
df_trade['rolling_mean_7'] = df_trade['price'].rolling(7).mean()
df_trade['rolling_std_7'] = df_trade['price'].rolling(7).std()
df_trade.dropna(inplace=True)

xtrade = df_trade[['lag_1', 'lag_5', 'lag_30', 'rolling_mean_7', 'rolling_std_7']].values
ytrade = df_trade['price'].values

tradepipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
param_grid_trade = {"ridge__alpha": [0.01, 0.1, 1, 10, 100]}  # double underscore!
grid_trade = GridSearchCV(tradepipe, param_grid=param_grid_trade, cv=TimeSeriesSplit(n_splits=5), scoring="r2")

grid_trade.fit(xtrade, ytrade)
print("best params", grid_trade.best_params_)
print("best score", grid_trade.best_score_)

# Bar chart: mean score per alpha
alphas = [0.01, 0.1, 1, 10, 100]
mean_scores = grid_trade.cv_results_['mean_test_score']
plt.bar([str(a) for a in alphas], mean_scores, color='teal')
plt.xlabel("Alpha")
plt.ylabel("Mean R² Score")
plt.title("GridSearchCV + TimeSeriesSplit: Score per Alpha")
plt.show()

"""
Full pipeline from scratch:
  1. Load california housing
  2. Learning curve plot → diagnose if model needs more data
  3. Validation curve → find best alpha range
  4. GridSearchCV → fine-tune best alpha
  5. Compare top 3 models with same CV folds
  6. Select best model with justification
  7. Final evaluation on held-out test set

Saara kaam ek script mein karo!
"""

# ===== Q6: FULL PIPELINE FROM SCRATCH =====
data = fetch_california_housing()
X_full, y_full = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# Step 1: Learning curve → diagnose
print("\n===== STEP 1: Learning Curve Diagnosis =====")
ts, tsc, vsc = learning_curve(Ridge(alpha=1), X_tr, y_tr, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='r2')
print(f"Final Train R²: {tsc.mean(axis=1)[-1]:.4f}, Final Val R²: {vsc.mean(axis=1)[-1]:.4f}")
print(f"Gap: {tsc.mean(axis=1)[-1] - vsc.mean(axis=1)[-1]:.4f}")
plt.plot(ts, tsc.mean(axis=1), label='Train')
plt.plot(ts, vsc.mean(axis=1), label='Validation')
plt.title('Learning Curve - Ridge'); plt.legend(); plt.grid(True); plt.show()

# Step 2: Validation curve → find best alpha range
print("\n===== STEP 2: Validation Curve =====")
alpha_range = [0.01, 0.1, 1, 10, 100, 500, 1000]
tsc2, vsc2 = validation_curve(Ridge(), X_tr, y_tr, param_name='alpha', param_range=alpha_range, cv=5, scoring='r2')
best_alpha_vc = alpha_range[np.argmax(vsc2.mean(axis=1))]
print(f"Best alpha from validation curve: {best_alpha_vc}")
plt.plot(alpha_range, tsc2.mean(axis=1), label='Train')
plt.plot(alpha_range, vsc2.mean(axis=1), label='Validation')
plt.xscale('log'); plt.title('Validation Curve'); plt.legend(); plt.grid(True); plt.show()

# Step 3: GridSearchCV → fine-tune
print("\n===== STEP 3: GridSearchCV Fine-Tune =====")
fine_alphas = np.logspace(np.log10(best_alpha_vc/10), np.log10(best_alpha_vc*10), 50)
grid_fine = GridSearchCV(Ridge(), {'alpha': fine_alphas}, cv=5, scoring='r2')
grid_fine.fit(X_tr, y_tr)
print(f"GridSearchCV best alpha: {grid_fine.best_params_['alpha']:.4f}")
print(f"GridSearchCV best R²: {grid_fine.best_score_:.4f}")

# Step 4: Compare top 3 models
print("\n===== STEP 4: Model Comparison =====")
kf_compare = KFold(n_splits=5, shuffle=True, random_state=42)
top3 = {
    'Ridge(best)': Ridge(alpha=grid_fine.best_params_['alpha']),
    'Lasso(0.1)': Lasso(alpha=0.1),
    'LinearReg': LinearRegression()
}
for name, m in top3.items():
    s = cross_val_score(m, X_tr, y_tr, cv=kf_compare, scoring='r2')
    print(f"{name:15s}: R² = {s.mean():.4f} ± {s.std():.4f}")

# Step 5: Final eval on held-out test
print("\n===== STEP 5: Final Test Set Evaluation =====")
best_model = Ridge(alpha=grid_fine.best_params_['alpha'])
best_model.fit(X_tr, y_tr)
test_r2 = best_model.score(X_te, y_te)
print(f"Final Test R²: {test_r2:.4f}")
print(f"Winner: Ridge(alpha={grid_fine.best_params_['alpha']:.4f})")

# ===== Q7: RandomizedSearchCV vs GridSearchCV Speed Test =====
print("\n===== Q7: Speed Comparison =====")
large_param = {
    'alpha': np.logspace(-3, 3, 100),  # 100 values: 0.001 to 1000
    'max_iter': [100, 500, 1000, 5000, 10000],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}
print(f"Total GridSearch combos: {100 * 5 * 7} = {100*5*7}")

# GridSearchCV
t1 = time.time()
grid_big = GridSearchCV(Ridge(), large_param, cv=3, scoring='r2')
grid_big.fit(X_tr, y_tr)
t_grid = time.time() - t1
print(f"\nGridSearchCV: {t_grid:.2f}s")
print(f"  Best: {grid_big.best_params_}")
print(f"  Score: {grid_big.best_score_:.4f}")

# RandomizedSearchCV
t2 = time.time()
rand_big = RandomizedSearchCV(Ridge(), large_param, n_iter=30, cv=3, scoring='r2', random_state=42)
rand_big.fit(X_tr, y_tr)
t_rand = time.time() - t2
print(f"\nRandomizedSearchCV: {t_rand:.2f}s")
print(f"  Best: {rand_big.best_params_}")
print(f"  Score: {rand_big.best_score_:.4f}")

print(f"\nSpeed difference: GridSearch {t_grid/t_rand:.1f}x SLOWER!")
print(f"Score difference: {abs(grid_big.best_score_ - rand_big.best_score_):.6f}")


"""
Is code mein 5 mistakes hain. Dhundh aur fix kar!
"""
# from sklearn.linear_model import Ridge
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import cross_val_score, GridSearchCV
# from sklearn.datasets import fetch_california_housing
# import numpy as np

# data = fetch_california_housing()
# X, y = data.data, data.target

# # Mistake 1 somewhere here
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# error hamne yaha par , pahle hi scale kar diya jo ki cheating haii , kyuki usne test data (jo future me hota ) uska bhi use karliya scalarisation kar ne me 

# # Mistake 2 somewhere here  
# from sklearn.model_selection import KFold
# kf = KFold(n_splits=5, shuffle=False)
#  isne shuffle = false karra joo ki galat hai , karna chahiye 



# # Mistake 3 somewhere here
# scores = cross_val_score(Ridge(), X_scaled, y, cv=kf, scoring='mean_squared_error')
#  isne ridge use kiya haaii jabki pipeline use karni thii 


# # Mistake 4 somewhere here
# from sklearn.pipeline import Pipeline
# pipe = Pipeline([('scaler', StandardScaler()), ('model', Ridge())])
# param_grid = {'alpha': [0.1, 1, 10]}
#  error isne "model_alpha" use nhi kiya hai 



# Mistake 5 somewhere here
# grid = GridSearchCV(pipe, param_grid, cv=5) isme error bass itna hi hai ki isne galat param use karre hai 
# grid.fit(X, y)
# print(f"Best: {grid.best_params_}") 