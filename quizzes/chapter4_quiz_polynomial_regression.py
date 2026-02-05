"""
================================================================================
CHAPTER 4 QUIZ SOLUTIONS - Polynomial Regression
================================================================================
Questions from: quiz_chapter_4_hard.md
Score: 32/35
================================================================================
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ============================================================================
# Q1: Design Matrix Construction (5 marks)
# Given: x = [1, 2, 3], y = [1, 8, 27] (y = x³)
# 1. Write the Design Matrix X for degree=3 polynomial (with intercept column)
# 2. What will be the shape of X?
# 3. If beta = (X^T @ X)^(-1) @ X^T @ Y, what shape will beta have?
# ============================================================================

x = pd.DataFrame([1, 2, 3])
Y = pd.DataFrame([1, 8, 27]) # (y = x³))
poly1=PolynomialFeatures(degree=3, include_bias=True)
X=poly1.fit_transform(x)
print(X)
print(X.shape)
beta = np.linalg.inv(X.T @ X) @ X.T @ Y
print("mere hisab se to beta ka sape hoga 4,1 , lets see ", beta.shape)


# ============================================================================
# Q2: Overfitting Analysis (6 marks)
# Given this table:
# | Degree | Train R² | Test R² |
# | 1      | 0.60     | 0.58    |
# | 2      | 0.85     | 0.83    |
# | 3      | 0.92     | 0.89    |
# | 5      | 0.97     | 0.75    |
# | 10     | 0.995    | 0.40    |
#
# 1. Which degree is overfitting? How do you know?
# 2. Which degree would you choose for production? Why?
# 3. What's the gap between Train R² and Test R² called?
# 4. How would you prevent overfitting?
# ============================================================================

#1. 10 degree is overfitting because we can see that test r2 iss faar diff than train r2
#2. degree 3 will be better bcs r2 is 92 which is nice but main thing is r2 of test is not that much diff than that of other degrees
#3.pata nhii.
# 4. by finding best degree where the test and train r2 is not much different

# ============================================================================
# Q3: Code Debugging (5 marks)
# Find 3 errors in this code:
# X = [1, 2, 3, 4, 5]  # Error 1: needs reshape
# y = [1, 4, 9, 16, 25]
# poly = PolynomialFeatures(degree=2)
# X_poly = poly.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_poly, y)  # Error 2: split before poly
# model = LinearRegression()
# model.fit(X_train, y_train)
# X_test_poly = poly.fit_transform(X_test)  # Error 3: should be transform() not fit_transform()
# y_pred = model.predict(X_test_poly)
# ============================================================================

X = np.array([1, 2, 3, 4, 5]).reshape(-1,1)
y =np.array([1, 4, 9, 16, 25]).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_train)
print(X_poly)

model = LinearRegression()

model.fit(X_poly, y_train)

# X_test_poly = poly.transform(X_test)
y_pred = model.predict(poly.transform(X_test))
print("b",model.intercept_)
'''[1.77635684e-15]'''
print("m",model.coef_)
'''[[0. 1.]]'''
print(y_pred)
'''[[ 4.]
 [25.]]'''

# ============================================================================
# Q4: Mathematical Understanding (6 marks)
# Given: y = 5 + 2x + 3x²
# 1. Is this model LINEAR or NON-LINEAR? Explain.
# 2. If we use matrix OLS: beta = (X^T @ X)^(-1) @ X^T @ Y, what will beta be?
# 3. Why can we use LINEAR regression to fit this CURVE?
# ============================================================================

# y = 5 + 2x + 3x²
# this is linear model but the line forming is not linear bcs of x^2 
# beta=[5,2,3]
# bcs it do not depend whether it is a curve or not , am i right ??

# ============================================================================
# Q5: Practical Scenario (8 marks)
# Temperature vs Ice Cream Sales data:
# | Temp | Sales |
# | 10   | 100   |
# | 20   | 400   |
# | 30   | 700   |
# | 40   | 600   |
#
# 1. Create polynomial features (degree 2)
# 2. Split data (80% train, 20% test)
# 3. Train LinearRegression
# 4. Print coefficients
# 5. Calculate R² on test set
# ============================================================================

tem=np.array([10,20,30,40]).reshape(-1,1)
sales=np.array([100,400,700,600]).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(tem,sales,test_size=0.2,random_state=42)
poly=PolynomialFeatures(degree=2,include_bias=False)
X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.transform(X_test)
model=LinearRegression()
model.fit(X_train_poly,y_train)
print("coefficiant haii ",model.coef_)
r2test=r2_score(y_test,model.predict(X_test_poly))
print(r2test)

# ============================================================================
# Q6: Transform vs Fit_Transform (5 marks)
# poly = PolynomialFeatures(degree=3)
# X_train_poly = poly.fit_transform(X_train)
# X_test_poly = poly.transform(X_test)  # Line A
# vs
# X_test_poly = poly.fit_transform(X_test)  # Line B
#
# 1. What's the difference between Line A and Line B?
# 2. Which one is CORRECT for test data?
# 3. What happens if you use the WRONG method?
# 4. Why does PolynomialFeatures need to "fit" at all?
# ============================================================================

poly = PolynomialFeatures(degree=3)
# X_train_poly = poly.fit_transform(X_train)
# X_test_poly = poly.transform(X_test)  # Line A
# vs
# X_test_poly = poly.fit_transform(X_test)  # Line B
# 1.  bhia sach bolu to mujhe koi diff nhi lagra haii kyuki end me dono hii [x,x^2] wali form dege too tu samjha
# 2. jaha tak mene sikha haii line a sahi hai
# 3. i  just know that poly will be learn that there are x no. of feature in test data and he has to use [x,x^2] , i think iff test data and train data have same feature and same shpe then there will be no problem , its my thinking i may be wrong tell me if it is 
# 4. broo u didmt tought mee that i ask the same q before , lekin me batana chahunga ki "fit " is used to know the shape or the number of feature the input data should have like if i fit x ,which has 2 feature then , when i'll do poly. transaform then it will required 2 feature .
