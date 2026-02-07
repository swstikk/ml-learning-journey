import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
np.random.seed(42)
X = np.sort(np.random.rand(20, 1) * 10, axis=0)
# print(X.shape)
# print(pd.DataFrame(X).head())
y = np.sin(X).ravel() + np.random.normal(0, 0.5, 20)
axes,fig=plt.subplots(2,2,figsize=(10,10))
deg=[1, 3, 10, 20]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

for i in deg:
    poly= PolynomialFeatures(degree=i, include_bias=False)
    poly_x=poly.fit_transform(x_train)
    
    poly_x_test=poly.fit_transform(x_test)
    ols=LinearRegression()
    ols.fit(poly_x,y_train)
    y_pred=ols.predict(poly_x)
    train_r2_score= r2_score(y_train,y_pred)# me dekhna chhata huu ki kaise me dono kaa use karke same r2 laa skata huu
    train_r2_score_copy= ols.score(poly_x,y_train)
    print(f"train r2 score copy : {train_r2_score_copy}")
    print(f"train r2 score: {train_r2_score}")
    test_score=ols.score(poly_x_test,y_test)
    print(f"test r2 score:b {test_score}")


    
    

    
