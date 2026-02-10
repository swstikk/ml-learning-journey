from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV ,KFold
fold=KFold(n_splits=5,shuffle=True,random_state=42)
house=fetch_california_housing()
param_gr={'model__alpha':[0.001,0.01,0.1,1,10,100]}  # model__ because step name is 'model'
x=house.data
y=house.target
pipe=Pipeline([('scalar',StandardScaler()),('model',Ridge())])
grid=GridSearchCV(pipe,param_gr,cv=fold)
grid.fit(x,y)

print(grid.best_params_)

print(grid.cv_results_['mean_test_score'])
print(grid.best_score_)
print(grid.best_estimator_)


# chalange 14
'''```
Stock data: Daily prices for 3 years (1095 days)
Goal: Predict next day price

a) TimeSeriesSplit with n_splits=10, test_size=30 use karo
b) Har split mein kitna training data hoga?
c) Gap=5 rakhne ka kya reason ho sakta hai?
d) Kyun regular KFold yahan galat hai?
```

a) First split training: 765 days
   Last split training:  1035 days

b) Training grows from 765 to 1035 days
   (increases by 30 days each split)

c) Gap=5 simulates:
   - Order execution delay
   - Predicting 5 days into future
   - More realistic trading scenario

d) Regular KFold problem:
   - Shuffle mixes past and future
   - Uses future data to predict past
   - IMPOSSIBLE in real trading
   - TimeSeriesSplit ensures chronological order'''
# chalange 15
'''Model ka CV output:
  Mean R² = 0.95
  Std = 0.02
  Train scores: [0.99, 0.99, 0.99, 0.99, 0.99]
  Test scores: [0.94, 0.96, 0.95, 0.94, 0.96]

a) Model kaisa perform kar raha hai?
b) Koi red flags hain?
c) Kya ye model production-ready hai?
d) Agar Train=0.99, Test=0.60 hota toh kya problem thi?



a)bohot badhiya hai 
b)nnhi haii koi red flag kyuki test me bhi badhiya kaam karra hai and std bhi accha haii 
c) hnn ready hai 
d) overfitting hota pakka fir yaa koi data leak .
vaise sach bolu to thoda sa shak hora hai ki kahi scalarisation karte vakt data leak to nhi ho gyaa test daata ke sath , agar nhi too shayad mene abhi itna accha prediction nhi dekha lol .
'''

