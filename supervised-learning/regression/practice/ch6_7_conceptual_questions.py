Tu same model 10 bar run karta hai different random_state ke sath:
Scores: [0.82, 0.91, 0.78, 0.88, 0.75, 0.93, 0.80, 0.85, 0.72, 0.89]

a) Ye variability kyun hai? 
b) Iska solution kya hai?
c) Agar tu sirf score 0.93 report kare toh kya problem hai?

answwer"
a kyuki ho sakta haii kabhi achha lucky data mil gya ho and voo kabhi lucky nikala jisse kabhi 0.91 to kabhi outliners ke vajah se 0.72"
b) iska solution hai cross validation yaa fir  par mujhe pata nhi kyuu kuch or bhi to ho sakta tha 
c) voo jhut hoga , i mean galat hoga kyuki ho skata hai voo data luckyy hoo . 

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
2: K-Fold Mechanics

Dataset: 100 samples, K=5

a) Har fold mein kitne training samples honge?
b) Har fold mein kitne test samples honge?
c) Total kitni baar model train hoga?
d) Ek sample kitni baar test set mein aayega?
e) Ek sample kitni baar training set mein aayega?

a) 80 samples hoge
b) 20 test samples hoge
c) 5 baar
d)1 baar
e) 4 baar

Q3: CV Score Analysis

5-Fold CV scores: [0.92, 0.45, 0.88, 0.50, 0.90]

a) Mean calculate karo
b) Std calculate karo
c) Kya ye model reliable hai? Kyun/Kyun nahi?
d) Ye high variance kya indicate kar raha hai?
e) Model mein kya problem ho sakti hai?

answwer:
a)73
b)np.std([0.92, 0.45, 0.88, 0.50, 0.90])
c) nhi kyuki std jayda aayas hai ,jyada variability hai iska mtlb model har baar same performance nhi dega 
d) model unstable haii ya galat model choose kiya hai 
e) model mee koi outliner ho skata hai yaa fir model overfit ho rha hai , iss ka reason tu thoda samjhaa
Q4: Choosing K Value

Scenario deduce karo:
a) n=30 samples → Kaunsa CV? loocv
b) n=500 samples → Kaunsa CV? 10 k fold 
c) n=50000 samples → Kaunsa CV? 5k fold
d) K badhane se bias/variance par kya effect hota hai? and bias kam hota haii and variance badhta haii , iska reason tu mujeh orr samjhaa 


Q5: Data Leakage (IMPORTANT!)

# Ye code mein kya galat hai?
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scores = cross_val_score(model, X_scaled, y, cv=5)
a) Problem identify karo 
b) Ye kyun "cheating" hai?
c) Correct implementation likho (Pipeline use karke)

answer: mean/ std hora hai vo test data ka bhi use akrra hai  joo cheating hai , .
solution hai 
from sklearn.model import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import linear_regression 
pipe=pipeline([('scaler',StandardScaler()),('model',linear_regression())])
scores=cross_val_score(pipe,X,y,cv=5)

CV Types Questions:
Q6: TimeSeriesSplit for Trading

Stock data: 365 days (1 year)
TimeSeriesSplit(n_splits=5)

a) First fold mein training size kya hogi approximately?
b) Last fold mein training size kya hogi?
c) Regular KFold se kya problem hoti trading mein?
d) test_size=30, max_train_size=180 set karne ka kya effect hoga?

answer:
mujhe nhi pata kaise calculate karre ki kya hoga pahla. 


Q8: Which CV to Use? (Decision Making)

Scenario batao konsa CV use karoge:
a) House price prediction (random data)
b) Stock price prediction (2020-2024 data)
c) Medical diagnosis with 40 samples
d) Customer churn prediction (5% churners, 95% non-churners)
e) Sensor data from 10 factories, multiple readings per factory


a) kfold
b) time series split
c) loocv
d) 
e) and bhai mujhe yee stratified and group cv nhi samjh aaya bilkul bhi 


Hyperparameter Tuning Questions:
Q9: GridSearchCV Calculation

param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100],      # 5 values
    'max_iter': [100, 1000, 5000],         # 3 values
    'solver': ['auto', 'svd', 'cholesky']  # 3 values
}
GridSearchCV(Ridge(), param_grid, cv=5)
a) Total kitne parameter combinations hain? 45
b) Kitne total model trainings honge? 225 (45*5)
c) Agar ek training mein 2 seconds lagein, total time? 450 sec 





























