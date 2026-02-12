import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Pipelines + Categorical Encoding — MEGA Practical File
========================================================
All Pipeline patterns + All Encoding methods + Real-world examples

Sections:
  1.  Pipeline Basics (Scale + Model)
  2.  Pipeline vs Manual (prove they give same result)
  3.  Pipeline Internal: What happens at each step
  4.  OneHotEncoder Deep Demo
  5.  OrdinalEncoder Demo
  6.  pd.get_dummies vs OneHotEncoder comparison
  7.  ColumnTransformer (mixed numeric + categorical)
  8.  SimpleImputer (missing values)
  9.  Full Production Pipeline
  10. GridSearchCV + Pipeline (tune everything)
  11. Data Leakage Demo (Pipeline prevents it!)
  12. Model Comparison with same Pipeline
  13. Real-World: Titanic-style dataset
  14. Pipeline Save/Load with joblib
  15. handle_unknown demo (new categories in test)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold
from sklearn.metrics import r2_score, accuracy_score
import time


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 1: PIPELINE BASICS                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

print("=" * 65)
print("  SECTION 1: Pipeline Basics (Scale + Model)")
print("=" * 65)

data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])

# Use like a normal model!
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
score = pipe.score(X_test, y_test)

print(f"\n  Pipeline: StandardScaler -> Ridge")
print(f"  Train Score: {pipe.score(X_train, y_train):.4f}")
print(f"  Test Score:  {score:.4f}")
print(f"  Pipeline steps: {[step[0] for step in pipe.steps]}")

# Access individual components
print(f"\n  Scaler mean (first 3): {pipe['scaler'].mean_[:3].round(3)}")
print(f"  Model coefs (first 3): {pipe['model'].coef_[:3].round(4)}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 2: PIPELINE vs MANUAL (Same Result!)                   ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n\n" + "=" * 65)
print("  SECTION 2: Pipeline vs Manual — SAME Result Proof")
print("=" * 65)

# Manual way
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
model = Ridge(alpha=1.0)
model.fit(X_train_s, y_train)
manual_score = model.score(X_test_s, y_test)
manual_pred = model.predict(X_test_s)

# Pipeline way
pipe = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))])
pipe.fit(X_train, y_train)
pipe_score = pipe.score(X_test, y_test)
pipe_pred = pipe.predict(X_test)

print(f"\n  Manual Score: {manual_score:.6f}")
print(f"  Pipe Score:   {pipe_score:.6f}")
print(f"  Predictions match: {np.allclose(manual_pred, pipe_pred)}")
print(f"  Coefficients match: {np.allclose(model.coef_, pipe['model'].coef_)}")
print(f"\n  PROVED: Pipeline = Manual, but cleaner and safer!")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 3: PIPELINE INTERNAL WORKING                           ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n\n" + "=" * 65)
print("  SECTION 3: Pipeline Internal — Step by Step")
print("=" * 65)

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])

print(f"\n  Pipeline has {len(pipe.steps)} steps:")
for i, (name, step) in enumerate(pipe.steps):
    step_type = "Transformer" if hasattr(step, 'transform') and i < len(pipe.steps)-1 else "Estimator"
    print(f"    Step {i+1}: {name} ({step.__class__.__name__}) [{step_type}]")

pipe.fit(X_train, y_train)

print(f"\n  When pipe.fit(X_train) is called:")
print(f"    1. imputer.fit_transform(X_train) -> X1")
print(f"    2. scaler.fit_transform(X1) -> X2")
print(f"    3. model.fit(X2, y_train)")
print(f"\n  When pipe.predict(X_test) is called:")
print(f"    1. imputer.transform(X_test) -> X1  (NO fit!)")
print(f"    2. scaler.transform(X1) -> X2       (NO fit!)")
print(f"    3. model.predict(X2) -> predictions")

# make_pipeline shortcut
pipe2 = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=1.0))
print(f"\n  make_pipeline auto-names: {[step[0] for step in pipe2.steps]}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 4: OneHotEncoder DEEP DEMO                              ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n\n" + "=" * 65)
print("  SECTION 4: OneHotEncoder — Deep Demo")
print("=" * 65)

# Simple example
colors = np.array([['Red'], ['Blue'], ['Green'], ['Red'], ['Blue']])
print(f"\n  Original data:\n  {colors.ravel()}")

# Basic OneHotEncoding
ohe = OneHotEncoder(sparse_output=False)
encoded = ohe.fit_transform(colors)
print(f"\n  After OneHotEncoder:")
print(f"  Categories found: {ohe.categories_[0]}")
print(f"  Feature names: {ohe.get_feature_names_out()}")
print(f"  Encoded:\n  {encoded}")
print(f"  Shape: {colors.shape} -> {encoded.shape} (1 col -> 3 cols!)")

# With drop='first' (avoid Dummy Variable Trap)
ohe_drop = OneHotEncoder(sparse_output=False, drop='first')
encoded_drop = ohe_drop.fit_transform(colors)
print(f"\n  With drop='first':")
print(f"  Feature names: {ohe_drop.get_feature_names_out()}")
print(f"  Encoded:\n  {encoded_drop}")
print(f"  Shape: {colors.shape} -> {encoded_drop.shape} (1 col -> 2 cols!)")
print(f"  'Blue' is the dropped (reference) category")

# Multiple categorical columns
multi_cat = np.array([
    ['Red', 'Small'], ['Blue', 'Large'], ['Green', 'Medium'],
    ['Red', 'Large'], ['Blue', 'Small']
])
ohe_multi = OneHotEncoder(sparse_output=False)
encoded_multi = ohe_multi.fit_transform(multi_cat)
print(f"\n  Multiple columns ({multi_cat.shape[1]}):")
print(f"  Feature names: {ohe_multi.get_feature_names_out()}")
print(f"  Shape: {multi_cat.shape} -> {encoded_multi.shape}")
print(f"  (2 cols -> {encoded_multi.shape[1]} cols! Each unique value = new column)")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 5: OrdinalEncoder DEMO                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n\n" + "=" * 65)
print("  SECTION 5: OrdinalEncoder — For Ordered Categories")
print("=" * 65)

sizes = np.array([['Large'], ['Small'], ['Medium'], ['XL'], ['Small']])
print(f"\n  Original: {sizes.ravel()}")

# Without specifying order (alphabetical default)
oe_default = OrdinalEncoder()
encoded_default = oe_default.fit_transform(sizes)
print(f"\n  Default OrdinalEncoder:")
print(f"  Categories: {oe_default.categories_[0]}")
print(f"  Encoded: {encoded_default.ravel()}")
print(f"  Problem! L=0, M=1, S=2, XL=3 -> Small(2) > Large(0)? WRONG!")

# With correct order specified
oe_ordered = OrdinalEncoder(categories=[['Small', 'Medium', 'Large', 'XL']])
encoded_ordered = oe_ordered.fit_transform(sizes)
print(f"\n  Ordered OrdinalEncoder:")
print(f"  Categories: {oe_ordered.categories_[0]}")
print(f"  Encoded: {encoded_ordered.ravel()}")
print(f"  Now: Small=0 < Medium=1 < Large=2 < XL=3 -> CORRECT!")

print(f"\n  When to use:")
print(f"    OrdinalEncoder: Size(S<M<L), Education(High School<BS<MS<PhD)")
print(f"    OneHotEncoder:  Color(no order), City(no order), Department(no order)")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 6: pd.get_dummies vs OneHotEncoder                     ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n\n" + "=" * 65)
print("  SECTION 6: pd.get_dummies vs OneHotEncoder")
print("=" * 65)

# Create sample dataframes
df_train = pd.DataFrame({'Color': ['Red', 'Blue', 'Green', 'Red'], 'Value': [10, 20, 30, 40]})
df_test = pd.DataFrame({'Color': ['Red', 'Blue', 'Yellow'], 'Value': [15, 25, 35]})  # Yellow is NEW!

print(f"\n  Train data: {df_train['Color'].tolist()}")
print(f"  Test data:  {df_test['Color'].tolist()}")
print(f"  'Yellow' is NEW in test (not in train)!")

# pd.get_dummies approach
dummies_train = pd.get_dummies(df_train, columns=['Color'])
dummies_test = pd.get_dummies(df_test, columns=['Color'])
print(f"\n  pd.get_dummies:")
print(f"  Train columns: {dummies_train.columns.tolist()}")
print(f"  Test columns:  {dummies_test.columns.tolist()}")
print(f"  PROBLEM: Test has 'Color_Yellow' but Train doesn't!")
print(f"  If you try to predict -> COLUMN MISMATCH ERROR!")

# OneHotEncoder approach
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(df_train[['Color']])
train_encoded = ohe.transform(df_train[['Color']])
test_encoded = ohe.transform(df_test[['Color']])
print(f"\n  OneHotEncoder (handle_unknown='ignore'):")
print(f"  Train encoded shape: {train_encoded.shape}")
print(f"  Test encoded shape:  {test_encoded.shape}")
print(f"  Test 'Yellow' row: {test_encoded[2]}  (all zeros! Safely handled)")
print(f"  SAME number of columns! No crash!")

print(f"\n  VERDICT: OneHotEncoder WINS for production ML!")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 7: ColumnTransformer (MIXED DATA)                      ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n\n" + "=" * 65)
print("  SECTION 7: ColumnTransformer — Mixed Numeric + Categorical")
print("=" * 65)

# Create realistic mixed dataset
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'Age': np.random.randint(20, 60, n).astype(float),
    'Salary': np.random.normal(50000, 15000, n),
    'Experience': np.random.randint(0, 30, n).astype(float),
    'City': np.random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Chennai'], n),
    'Education': np.random.choice(['BSc', 'MSc', 'PhD'], n),
    'Department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], n),
})
# Target: performance score
y_mix = (0.3 * df['Age'] + 0.0001 * df['Salary'] + 0.5 * df['Experience'] + 
         np.random.normal(0, 5, n))

num_cols = ['Age', 'Salary', 'Experience']
cat_cols = ['City', 'Education', 'Department']

print(f"\n  Dataset: {df.shape}")
print(f"  Numeric columns: {num_cols}")
print(f"  Categorical columns: {cat_cols}")
print(f"\n  Sample data:")
print(f"  {df.head(3).to_string()}")

# ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
])

X_processed = preprocessor.fit_transform(df)
print(f"\n  After ColumnTransformer:")
print(f"  Original shape: {df.shape}")
print(f"  Processed shape: {X_processed.shape}")
print(f"  Numeric features: {len(num_cols)} (scaled)")

cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
print(f"  Encoded features ({len(cat_feature_names)}): {cat_feature_names.tolist()}")
print(f"  Total: {len(num_cols)} numeric + {len(cat_feature_names)} encoded = {X_processed.shape[1]}")

# Full Pipeline with model
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=1.0))
])

X_tr, X_te, y_tr, y_te = train_test_split(df, y_mix, test_size=0.2, random_state=42)
pipe.fit(X_tr, y_tr)
print(f"\n  Full Pipeline Score:")
print(f"  Train R2: {pipe.score(X_tr, y_tr):.4f}")
print(f"  Test R2:  {pipe.score(X_te, y_te):.4f}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 8: SimpleImputer (Missing Values)                      ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n\n" + "=" * 65)
print("  SECTION 8: SimpleImputer — Handle Missing Values")
print("=" * 65)

# Create data with missing values
df_missing = pd.DataFrame({
    'Age': [25, np.nan, 35, 28, np.nan, 42, 30, np.nan, 50, 33],
    'Salary': [30000, 50000, np.nan, 35000, 80000, np.nan, 55000, 40000, np.nan, 60000],
    'City': ['Delhi', 'Mumbai', np.nan, 'Delhi', 'Mumbai', 'Bangalore', np.nan, 'Delhi', 'Mumbai', np.nan],
    'Score': [65, 78, 85, 70, 92, 88, 72, 68, 95, 75]
})

print(f"\n  Data with missing values:")
print(f"  {df_missing.to_string()}")
print(f"\n  Missing count:")
print(f"  {df_missing.isnull().sum().to_string()}")

# Numeric imputation strategies
for strategy in ['mean', 'median', 'constant']:
    imp = SimpleImputer(strategy=strategy, fill_value=0 if strategy == 'constant' else None)
    filled = imp.fit_transform(df_missing[['Age', 'Salary']])
    age_fill = filled[:, 0]
    ages_that_were_nan = [age_fill[1], age_fill[4], age_fill[7]]
    print(f"\n  Strategy '{strategy}':")
    print(f"    Missing Ages filled with: {[round(float(x), 1) for x in ages_that_were_nan]}")

# Categorical imputation
cat_imp = SimpleImputer(strategy='most_frequent')
filled_city = cat_imp.fit_transform(df_missing[['City']])
print(f"\n  Categorical (most_frequent):")
print(f"    Most frequent city: {cat_imp.statistics_[0]}")
print(f"    Missing cities filled with: '{cat_imp.statistics_[0]}'")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 9: FULL PRODUCTION PIPELINE                             ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n\n" + "=" * 65)
print("  SECTION 9: Full Production Pipeline (Missing + Mixed)")
print("=" * 65)

# Create realistic messy dataset with missing values
np.random.seed(42)
n = 500
df_real = pd.DataFrame({
    'Age': np.where(np.random.random(n) > 0.9, np.nan, np.random.randint(20, 60, n).astype(float)),
    'Income': np.where(np.random.random(n) > 0.85, np.nan, np.random.normal(60000, 20000, n)),
    'Experience': np.where(np.random.random(n) > 0.95, np.nan, np.random.randint(0, 35, n).astype(float)),
    'City': np.where(np.random.random(n) > 0.9, None, 
                     np.random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata'], n)),
    'Education': np.where(np.random.random(n) > 0.88, None,
                         np.random.choice(['High School', 'BSc', 'MSc', 'PhD'], n)),
})
y_real = (0.4 * df_real['Age'].fillna(35) + 0.0001 * df_real['Income'].fillna(50000) + 
          0.6 * df_real['Experience'].fillna(10) + np.random.normal(0, 5, n))

num_features = ['Age', 'Income', 'Experience']
cat_features = ['City', 'Education']

print(f"\n  Messy dataset: {df_real.shape}")
print(f"  Missing values:")
print(f"  {df_real.isnull().sum().to_string()}")

# Build the FULL production pipeline
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features),
])

full_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=1.0))
])

# Split and train
X_tr, X_te, y_tr, y_te = train_test_split(df_real, y_real, test_size=0.2, random_state=42)

full_pipe.fit(X_tr, y_tr)
print(f"\n  Full Production Pipeline:")
print(f"  Train R2: {full_pipe.score(X_tr, y_tr):.4f}")
print(f"  Test R2:  {full_pipe.score(X_te, y_te):.4f}")

# Cross-validation with messy data — Pipeline handles EVERYTHING
cv_scores = cross_val_score(full_pipe, df_real, y_real, cv=5, scoring='r2')
print(f"  CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"  No errors despite missing values! Pipeline handled it all!")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 10: GridSearchCV + PIPELINE                             ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n\n" + "=" * 65)
print("  SECTION 10: GridSearchCV + Pipeline — Tune Everything!")
print("=" * 65)

# Tune model alpha
param_grid = {
    'model__alpha': [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(full_pipe, param_grid, cv=5, scoring='r2', return_train_score=True)
grid.fit(X_tr, y_tr)

print(f"\n  GridSearchCV Results:")
print(f"  {'Alpha':<10} {'Train R2':>10} {'CV R2':>10}")
print(f"  {'-'*32}")
for i, alpha in enumerate(param_grid['model__alpha']):
    train_s = grid.cv_results_['mean_train_score'][i]
    test_s = grid.cv_results_['mean_test_score'][i]
    marker = " <-- BEST" if alpha == grid.best_params_['model__alpha'] else ""
    print(f"  {alpha:<10} {train_s:>10.4f} {test_s:>10.4f}{marker}")

print(f"\n  Best params: {grid.best_params_}")
print(f"  Best CV R2: {grid.best_score_:.4f}")
print(f"  Test R2:    {grid.score(X_te, y_te):.4f}")

# ADVANCED: Tune imputer strategy + model alpha
print(f"\n  ADVANCED: Tune imputer strategy too!")
param_grid_adv = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'model__alpha': [0.1, 1, 10]
}

grid_adv = GridSearchCV(full_pipe, param_grid_adv, cv=5, scoring='r2')
grid_adv.fit(X_tr, y_tr)
print(f"  Best params: {grid_adv.best_params_}")
print(f"  Best CV R2:  {grid_adv.best_score_:.4f}")
print(f"\n  Parameter path explanation:")
print(f"  preprocessor ── num ── imputer ── strategy")
print(f"  (step1)    (substep) (substep)  (param)")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 11: DATA LEAKAGE DEMO                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n\n" + "=" * 65)
print("  SECTION 11: Data Leakage — Pipeline Prevents It!")
print("=" * 65)

# Numerical-only for clarity
X_cal = data.data
y_cal = data.target
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_cal, y_cal, test_size=0.2, random_state=42)

# LEAKY: Scale ALL data, then CV
scaler_leak = StandardScaler()
X_leaked = scaler_leak.fit_transform(X_cal)  # fit on ALL data!
scores_leak = cross_val_score(Ridge(alpha=10), X_leaked, y_cal, cv=5, scoring='r2')

# CORRECT: Pipeline (scale inside CV)
pipe_safe = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=10))])
scores_safe = cross_val_score(pipe_safe, X_cal, y_cal, cv=5, scoring='r2')

print(f"\n  LEAKY (scale all, then CV):    R2 = {scores_leak.mean():.4f} +/- {scores_leak.std():.4f}")
print(f"  SAFE  (Pipeline, scale in CV): R2 = {scores_safe.mean():.4f} +/- {scores_safe.std():.4f}")
print(f"  Difference: {abs(scores_leak.mean() - scores_safe.mean()):.4f}")
print(f"\n  LEAKY score is slightly INFLATED (looks better than real)")
print(f"  The Pipeline score is the TRUE performance!")
print(f"  In small datasets, this difference can be HUGE!")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 12: MODEL COMPARISON WITH SAME PIPELINE                ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n\n" + "=" * 65)
print("  SECTION 12: Compare Models with Same Preprocessing")
print("=" * 65)

# Same preprocessor, different models
models = {
    'Ridge(1)': Ridge(alpha=1),
    'Ridge(10)': Ridge(alpha=10),
    'Lasso(0.1)': Lasso(alpha=0.1),
    'DecisionTree(d=8)': DecisionTreeRegressor(max_depth=8, random_state=42),
    'KNN(5)': KNeighborsRegressor(n_neighbors=5),
}

print(f"\n  {'Model':<22} {'CV R2':>10} {'Time (s)':>10}")
print(f"  {'-'*45}")

for name, model in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    t = time.time()
    scores = cross_val_score(pipe, df_real, y_real, cv=5, scoring='r2')
    t = time.time() - t
    print(f"  {name:<22} {scores.mean():>10.4f} {t:>10.3f}")

print(f"\n  Same preprocessing, fair comparison!")
print(f"  Change model in ONE line, rest stays same!")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 13: REAL-WORLD STYLE DATASET                           ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n\n" + "=" * 65)
print("  SECTION 13: Real-World Style — Employee Dataset")
print("=" * 65)

# Create a Titanic-style dataset
np.random.seed(42)
n = 1000
df_emp = pd.DataFrame({
    'Age': np.where(np.random.random(n) > 0.92, np.nan, np.random.randint(22, 58, n).astype(float)),
    'YearsExp': np.where(np.random.random(n) > 0.95, np.nan, np.random.randint(0, 30, n).astype(float)),
    'HoursPerWeek': np.random.randint(20, 60, n).astype(float),
    'Salary': np.where(np.random.random(n) > 0.9, np.nan, np.random.normal(55000, 18000, n)),
    'Department': np.where(np.random.random(n) > 0.95, None,
                          np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR', 'Finance'], n)),
    'Education': np.where(np.random.random(n) > 0.93, None,
                         np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n)),
    'City': np.where(np.random.random(n) > 0.97, None,
                    np.random.choice(['NYC', 'SF', 'Chicago', 'Austin', 'Seattle', 'Boston'], n)),
})

# Target: Promoted or not (binary classification!)
promo_prob = (0.02 * df_emp['Age'].fillna(35) + 
              0.03 * df_emp['YearsExp'].fillna(10) +
              0.01 * df_emp['HoursPerWeek'] - 1.5)
promo_prob = 1 / (1 + np.exp(-promo_prob))  # sigmoid
y_promo = (np.random.random(n) < promo_prob).astype(int)

print(f"\n  Dataset: {df_emp.shape}")
print(f"  Target (Promoted): {y_promo.mean():.1%} positive rate")
print(f"  Missing values:\n  {df_emp.isnull().sum().to_string()}")

# Auto-detect columns
num_cols_auto = df_emp.select_dtypes(include='number').columns.tolist()
cat_cols_auto = df_emp.select_dtypes(include='object').columns.tolist()
print(f"\n  Auto-detected numeric: {num_cols_auto}")
print(f"  Auto-detected categorical: {cat_cols_auto}")

# Full classification pipeline!
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
])

preprocessor_class = ColumnTransformer([
    ('num', num_pipe, num_cols_auto),
    ('cat', cat_pipe, cat_cols_auto),
])

# Compare classification models!
classifiers = {
    'LogisticReg': LogisticRegression(max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'KNN(5)': KNeighborsClassifier(n_neighbors=5),
}

print(f"\n  Classification Results (5-fold CV Accuracy):")
print(f"  {'Model':<18} {'Accuracy':>10} {'Std':>10}")
print(f"  {'-'*40}")

for name, clf in classifiers.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor_class),
        ('model', clf)
    ])
    scores = cross_val_score(pipe, df_emp, y_promo, cv=5, scoring='accuracy')
    print(f"  {name:<18} {scores.mean():>10.4f} {scores.std():>10.4f}")

# GridSearch on best model
print(f"\n  GridSearchCV on LogisticRegression:")
best_pipe = Pipeline([
    ('preprocessor', preprocessor_class),
    ('model', LogisticRegression(max_iter=1000))
])

param_grid_class = {
    'model__C': [0.01, 0.1, 1, 10],
    'model__solver': ['lbfgs', 'liblinear']
}

grid_class = GridSearchCV(best_pipe, param_grid_class, cv=5, scoring='accuracy')
X_tr_e, X_te_e, y_tr_e, y_te_e = train_test_split(df_emp, y_promo, test_size=0.2, random_state=42)
grid_class.fit(X_tr_e, y_tr_e)

print(f"  Best params: {grid_class.best_params_}")
print(f"  Best CV Accuracy: {grid_class.best_score_:.4f}")
print(f"  Test Accuracy:    {grid_class.score(X_te_e, y_te_e):.4f}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 14: SAVE AND LOAD PIPELINE                              ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n\n" + "=" * 65)
print("  SECTION 14: Save and Load Pipeline with joblib")
print("=" * 65)

import joblib
import os

# Save
save_path = os.path.join(os.path.dirname(__file__), 'saved_pipeline.pkl')
joblib.dump(grid_class.best_estimator_, save_path)
print(f"\n  Saved pipeline to: {save_path}")
print(f"  File size: {os.path.getsize(save_path) / 1024:.1f} KB")

# Load
loaded_pipe = joblib.load(save_path)
loaded_pred = loaded_pipe.predict(X_te_e)
original_pred = grid_class.predict(X_te_e)

print(f"  Predictions match after load: {np.array_equal(loaded_pred, original_pred)}")
print(f"  Loaded pipe accuracy: {loaded_pipe.score(X_te_e, y_te_e):.4f}")
print(f"\n  ONE file = scaler + imputer + encoder + model!")
print(f"  Deploy to production with just: joblib.load() -> predict()")

# Cleanup
os.remove(save_path)
print(f"  (Cleaned up saved file)")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 15: handle_unknown DEMO                                ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n\n" + "=" * 65)
print("  SECTION 15: handle_unknown — New Categories in Test Data")
print("=" * 65)

# Train only has Delhi, Mumbai, Bangalore
train_cities = np.array([['Delhi'], ['Mumbai'], ['Bangalore'], ['Delhi'], ['Mumbai']])
# Test has Kolkata (NEW!)
test_cities = np.array([['Delhi'], ['Kolkata'], ['Mumbai']])

# Without handle_unknown -> ERROR
try:
    ohe_strict = OneHotEncoder(sparse_output=False)
    ohe_strict.fit(train_cities)
    ohe_strict.transform(test_cities)
    print(f"\n  Without handle_unknown: Success")
except ValueError as e:
    print(f"\n  Without handle_unknown: ERROR!")
    print(f"  {str(e)[:80]}...")

# With handle_unknown='ignore' -> SAFE
ohe_safe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_safe.fit(train_cities)
result = ohe_safe.transform(test_cities)
print(f"\n  With handle_unknown='ignore':")
print(f"  Categories: {ohe_safe.categories_[0]}")
print(f"  Delhi:    {result[0]}")
print(f"  Kolkata:  {result[1]}  <- all zeros! (unknown category)")
print(f"  Mumbai:   {result[2]}")
print(f"\n  Unknown category = all-zero row (safely ignored)")
print(f"  Model will make predictions as if no city was specified")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  FINAL SUMMARY                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)

print("""
  Pipeline Patterns:
  1. Simple:      StandardScaler -> Model
  2. Imputer:     SimpleImputer -> StandardScaler -> Model
  3. Mixed:       ColumnTransformer(num/cat) -> Model
  4. Production:  ColumnTransformer(Imputer+Scale / Imputer+Encode) -> Model

  Encoding:
  - OneHotEncoder: No order (Color, City) -> multiple binary columns
  - OrdinalEncoder: Has order (Size, Education) -> single integer column
  - pd.get_dummies: AVOID in ML pipelines (use OneHotEncoder instead!)

  Key Rules:
  - Pipeline prevents data leakage automatically
  - Use handle_unknown='ignore' for OneHotEncoder
  - Use drop='first' for Linear/Logistic Regression
  - Parameter naming: step__substep__param (double underscores)
  - Save entire Pipeline with joblib (one file for everything)
""")

print("=" * 65)
print("  DONE! Pipelines + Encoding COMPLETE!")
print("  15 sections, all patterns covered!")
print("=" * 65)
