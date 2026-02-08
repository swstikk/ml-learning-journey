
import numpy as np 
from sklearn.metrics import mean_squared_error,r2_score
'''Q1: You have 2 models:

Model A: RMSE = 5.2, R² = 0.85
Model B: RMSE = 7.1, R² = 0.91
Which is better? Why might this happen?'''

'''  
model A is better because it has lower RMSE and higher R²,
 soo we can see that it has higher vairance but lower bias , that means it may be over fitted'''

'''Q2: Your model gives:

Train R² = 0.95
Test R² = 0.40
What's the problem? How to fix?'''
'''again the problem is that the model is over fitted , because it has higher R² on training data and lower R² on testing data , 
we can fix this by using regularization , cross validation , or reducing the number of feature s using lasso cv , it will remove unwanter features'''
#  ye pakaa model jyada complex hone ki vajah sse hua hai .

'''Q3: Calculate by hand:

True:      [5, 10, 15]
Predicted: [6, 9, 18]
Find: MSE, RMSE, MAE'''
'''(1+1+9)/3=3.6666=mse
rmse=np.sqrt(mse)=1.9148
mae=(1+1+3)/3=1.666'''

# Q4: When would you prefer MAE over RMSE?
# when the outliner is present in data bcs rmse will react more on outliner but mae will not 

# Q5: Can Adjusted R² ever be higher than R²? Why/why not?
''' let me give u answer in mathematics form 
r2=1-ss_res/ss_tot              eq1 
so
ss_res/ss_tot=1-r2

since,
Adj R² = 1 - [(1 - r2) * (n - 1) / (n - p - 1)]
Adj R² = 1 - [(ss_res/ss_tot) * (n - 1) / (n - p - 1)]    eq2

now compare eq1 and eq2 :
Adj R² = 1 - [(ss_res/ss_tot) * (n - 1) / (n - p - 1)]   
r2     = 1 - [(ss_res/ss_tot)                        ] (mene bass iase hi baracket laga diye compare ke liey )

it is clearly visible that adj r2 is always less than r2 or
max to max equal to r2 only when parameter is 0 

Where:
  n = number of samples
  p = number of features

If p = 0 (no features), then:
Adj R² = 1 - [(1 - R²) * (n - 1) / (n - 1)]
Adj R² = 1 - (1 - R²)
Adj R² = R²

So, Adjusted R² can never be higher than R²!'''

# '''Q6: A model has Test R² = -0.3. What does this mean?
#  may be the model iss overfitted(complex_model) orr model is not suitable for the data (may be it need polynomial feature).

'''Q7: Two datasets:

Dataset A: House prices ($100k-$1M), RMSE = 50,000
Dataset B: Car prices ($10k-$50k), RMSE = 15,000
Which model is better?'''
# obviously model a is better , i dont know how to calculate that but we can see that despite of that much big numbers it only giving mse  2500000000 while house data was giving 225000000 


# ...............................................................................
# 🔥 Coding Challenges
# Challenge 1: Implement Metrics from Scratch
# WITHOUT using sklearn!
def calculate_mse(y_true, y_pred):
    
    return np.sum((y_true-y_pred)**2)
    # Your code here
   

def calculate_r2(y_true, y_pred):
    return 1-calculate_mse(y_true,y_pred)/calculate_mse(y_true,np.mean(y_true))

# Test:
y_true = np.array([10, 20, 30, 40])
y_pred = np.array([12, 18, 35, 38])

print(r2_score(y_true,y_pred))
print(calculate_r2(y_true,y_pred))
# Should match sklearn output!
# ...............................................................................
# Challenge 2: Negative R² Detector
# Write a function that:
print("Challenge 2: Negative R² Detector")
# Takes train/test R² scores
# Diagnoses: Overfit? Underfit? Good fit?
# Returns recommendations
def diagnose_model(train_r2, test_r2):
    if train_r2>0.9 and test_r2<0.7:
        return "overfitted","use regularisation"
    elif train_r2<=0.5 and test_r2<=0.5:
        return "underfitted","use another model"
    else:
        return "good fit", "eat 5 star"
    # Your code here
    # Return: diagnosis (str) and recommendation (str)
   

# Test cases:
print(diagnose_model(0.95, 0.40))  # Should detect overfit
print(diagnose_model(0.50, 0.48))  # Should detect underfit
print(diagnose_model(0.85, 0.82))  # Should say good!

# ...............................................................................
# Challenge 3: Best Model Selector
# Given 3 models with different metrics, select the best one.
# Consider: RMSE, R², and train/test gap!

print("Challenge 3: Best Model Selector")


def select_best_model(models):
    """
    models = list of dicts with keys:
        'name', 'train_r2', 'test_r2', 'test_rmse'
    
    Selection criteria:
    1. Reject overfitting models (train_r2 - test_r2 > 0.15)
    2. Among remaining, pick lowest test_rmse
    3. If tie, pick highest test_r2
    """
    valid_models = []
    
    print("\nAnalyzing models...")
    for m in models:
        gap = m['train_r2'] - m['test_r2']
        
        if gap > 0.15:
            print(f"    Status: REJECTED (Overfitting! Gap > 0.15)")
        elif m['test_r2'] < 0.5:
            print(f"    Status: REJECTED (Underfitting! Test R² < 0.5)")
        else:
            print(f"    Status: VALID")
            valid_models.append(m)
    
    if not valid_models:
        return None, "All models are bad! Need better approach."
    
    # Sort by test_rmse (ascending), then by test_r2 (descending)
    valid_models.sort(key=lambda x: (x['test_rmse'], -x['test_r2']))
    best = valid_models[0]
    
    return best['name'], f"Lowest RMSE ({best['test_rmse']:.2f}) with good R² ({best['test_r2']:.3f})"

# Test with 3 models
models = [
    {'name': 'Linear Regression', 'train_r2': 0.85, 'test_r2': 0.82, 'test_rmse': 15.2},
    {'name': 'Polynomial Deg 10', 'train_r2': 0.99, 'test_r2': 0.65, 'test_rmse': 22.5},  # Overfit!
    {'name': 'Ridge Regression', 'train_r2': 0.88, 'test_r2': 0.86, 'test_rmse': 12.8},
]

best_name, reason = select_best_model(models)

print(f"WINNER: {best_name}")
print(f"Reason: {reason}")



bad_models = [
    {'name': 'Overfit Model', 'train_r2': 0.99, 'test_r2': 0.30, 'test_rmse': 50.0},
    {'name': 'Underfit Model', 'train_r2': 0.40, 'test_r2': 0.38, 'test_rmse': 40.0},
]
best_name, reason = select_best_model(bad_models)
print(f"\nResult: {best_name}")
print(f"Reason: {reason}")