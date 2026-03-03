from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.datasets import load_diabetes
X,y=load_diabetes(return_X_y=True)
print(X.shape)
print(y.shape)
params={
"poly__degree": [1, 2, 3],
"ridge__alpha": [0.1, 1, 10, 100, 1000]}
print("total fits= 3*5*5=75")
pipe=Pipeline([
    ('scalar',StandardScaler()),
    ('poly',PolynomialFeatures()),
    ('ridge',Ridge())
])
gd= GridSearchCV(pipe,param_grid=params, cv=5,scoring='r2')
gd.fit(X,y)

print(gd.best_params_) # poly =1 , ridge =0.1
mm=cross_val_score(Ridge(alpha=0.1),X,y,cv=5)
print(mm.mean())#0.47988210231953665
print(gd.best_score_)#
ols= LinearRegression()
ols.fit(X,y)
print(ols.score(X,y))
print("ols(simple linear model is better than polynomial or ridge regression )")