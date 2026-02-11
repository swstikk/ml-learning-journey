from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, RidgeCV, LassoCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

import numpy as np 

ols= LinearRegression()
x, y, coef = make_regression(n_samples=100, n_features=100, n_informative=3, noise=10, coef=True, random_state=42)

X_quad = np.linspace(-10, 10, 100).reshape(-1, 1)
y_quad = X_quad**2 + np.random.normal(0, 5, X_quad.shape)

alphas = np.logspace(-1,1.5,100)
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')

ridge_cv.fit(x, y)

print(f"Best Alpha found: {ridge_cv.alpha_}")
print(f"Coefficients: {ridge_cv.coef_}")
 

'''Does increasing Ridge alpha help the model fit the curve? Or does it just make it flatter (high bias)?'''
# answer= nhi kar sakta haii ,
#  ridge sirf coef_ ko choota karti haii---> 
# varaince choota hoga---> 
# stable banega ----> 
# but agar model hi galat haii too ---> 
# prediction hii galat hai too bias kabhi fix nhi  hogaa .
# .............................................
#  jaise bhia tum linear lelo where true predictiuon is y=x^2 ,
#  tu kitna bhi coeef ke sath khel loo 
# ,kitna bhi maje lelo ,
#  tum kabhi usska bias fix kar nhi paoge 
#  believe nhi haii to ex ko bhejh do apni (usse achha koi nhi kehl sakta )

# now 
poly = PolynomialFeatures(degree=2)
poly_x=poly.fit_transform(X_quad)
ols.fit(poly_x,y_quad)
plt.scatter(X_quad,y_quad,label="Actual")
plt.plot(X_quad,ols.predict(poly_x),color="red",label="Predicted")
plt.legend()
plt.show()










