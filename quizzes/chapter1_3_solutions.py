"""
================================================================================
CHAPTER 1-3 QUIZ SOLUTIONS - Linear Regression
================================================================================
Questions from: quiz_chapter_1_to_3.md
My solutions and practice code
================================================================================
"""

# ============================================================================
# Q1: OLS Formula (5 marks)
# Using only ONE feature (MedInc), calculate slope (m) and intercept (b) manually.
# m = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
# b = ȳ - m·x̄
# ============================================================================

import numpy as np 
import pandas as pd 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
data= fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names )
df["Price"]=data.target
df=df.sample(n=500,random_state=42)
X = df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms']]
# print(data.feature_names)  # Just 3 features
Y = df['Price']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model= LinearRegression()
model.fit(X_train,Y_train)
predict=model.predict(X_test)

# ============================================================================
# Q2: R² Interpretation (3 marks)
# What does R² value mean?
# If R² = 0.4, what percentage of variance is UNEXPLAINED?
# ============================================================================

r2= 1- np.sum((Y_test-predict)**2)/np.sum((Y_test-np.mean(Y_test))**2)
# or pata nhi mee bhul gya ki r2 library se kaise nikal te haii
print(model.score(X_test,Y_test),r2)

# ============================================================================
# Q3: Design Matrix (5 marks)
# Write the Design Matrix (X) with One-Hot Encoding (drop_first=True)
# Given: Size=[10,20,15], Color=[Red,Blue,Red], Price=[100,200,150]
# ============================================================================

df_q3=pd.DataFrame({"Size":[10,20,15],"Color":["Red","Blue","Red"],"Price":[100,200,150]})
print(df_q3)
print (pd.get_dummies(df_q3,drop_first= True))
print (pd.get_dummies(df_q3,drop_first= False))

# ============================================================================
# Q4: Coefficient Interpretation (4 marks)
# 1. If MedInc coefficient is +0.45, what does it mean?
# 2. If a coefficient is negative, what does that indicate?
# 3. Which feature has the MOST impact on price?
# ============================================================================

vif=pd.DataFrame()
vif["Features"]=X.columns
# vif["VIF"]=variance_inflation_factor(X.values,range(len(X.columns)  # sklearn ka use karke vif nikalna hai 
rmse=np.sqrt(np.mean((Y_test-predict)**2))

# ============================================================================
# Q5: Assumptions Check (6 marks)
# 1. What DW value indicates good independence? Is this model OK?
# 2. What does VIF > 10 mean? 
# 3. If VIF for AveRooms = 50, what would you do?
# ============================================================================

print(Y_test)

# ============================================================================
# Q6: Residual Plot Analysis (5 marks)
# 1. What pattern indicates linearity assumption is MET?
# 2. What pattern indicates heteroscedasticity?
# 3. What would you do if you see a U-shaped curve?
# ============================================================================

plt.scatter(Y_test-predict,Y_test)

plt.xlabel("Actual")
plt.ylabel("Residual")
plt.show()

# coefficients check
print(np.abs(model.coef_))
