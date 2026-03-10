# Chapter 1: Logistic Regression — Poori Kahani, Step by Step

> **Pehle samjho KYUN chahiye, phir KYA hai, phir KAISE kaam karta hai.**
> **Math level: Class 11 (exponents, log rules, basic differentiation)**

---

## PART 1: Problem Kya Hai? (Kyun Classification Chahiye?)

### Regression Yaad Hai?

Linear Regression mein hum ek **number** predict karte the:
```
Input: Ghar ka size (1200 sq ft)
Output: Price = ₹25,00,000     ← ye ek continuous number hai
```

Lekin real life mein bahut saare problems hain jahan answer **number nahi, category** hai:

```
Doctor:     Blood report dekh ke batao → Cancer hai ya nahi?     → YES / NO
Bank:       Ye transaction fraud hai ya nahi?                    → FRAUD / LEGIT
Email:      Ye email spam hai ya real?                           → SPAM / NOT SPAM
Biology:    Is gene ka expression disease karega ya nahi?        → DISEASE / HEALTHY
```

**Yahan answer 0 ya 1 mein aata hai. Isko bolte hain CLASSIFICATION.**

### Linear Regression Kyun FAIL Hota Hai Classification Mein?

Soch, agar hum Linear Regression lagayen cancer prediction pe:

```
Data:
  Patient 1: Tumor Size = 1 cm  → Cancer = 0 (No)
  Patient 2: Tumor Size = 2 cm  → Cancer = 0 (No)
  Patient 3: Tumor Size = 3 cm  → Cancer = 0 (No)
  Patient 4: Tumor Size = 5 cm  → Cancer = 1 (Yes)
  Patient 5: Tumor Size = 7 cm  → Cancer = 1 (Yes)
  Patient 6: Tumor Size = 8 cm  → Cancer = 1 (Yes)
```

Agar hum straight line fit karein (Linear Regression):

```
   Cancer?
    1.0 |                          * * *
        |                    ╱─────
        |              ╱─────
    0.5 |        ╱─────           ← Yahan se upar = Cancer?
        |  ╱─────
    0.0 |* * *
   -0.5 |  ← YE KYA HAI? -0.5 probability? 
        +--+--+--+--+--+--+--+--→ Tumor Size (cm)
```

**3 PROBLEMS:**

**Problem 1: Output range galat hai**
- Linear Regression `-∞ se +∞` tak predict karta hai
- Lekin probability `0% se 100%` ke beech honi chahiye
- Agar model bole "probability = 1.5" → iska matlab kya? 150% chance? MEANINGLESS!

**Problem 2: Outlier effect**
- Agar ek patient aaye jiska Tumor Size = 50 cm
- Poori line shift ho jaayegi
- Jo pehle sahi predict ho raha tha, wo galat ho jaayega

**Problem 3: Threshold issue**
- Hum 0.5 pe cut lagayen? Lekin line straight hai, toh kahin bhi ek chota
  sa change poori prediction badal deta hai

**SOLUTION KYA CHAHIYE:**
```
Hume ek aisa function chahiye jo:
  ✅ Koi bhi input lo (-∞ to +∞)
  ✅ Output HAMESHA 0 se 1 ke beech ho (probability)
  ✅ S-shape ka ho (smoothly 0 se 1 pe jaaye)
  
Ye function hai → SIGMOID FUNCTION!
```

---

## PART 2: Sigmoid Function — Heart of Logistic Regression

### Pehle Samjho `e` Kya Hai (Quick Refresher)

`e` ek special number hai math mein:
```
e ≈ 2.71828...

Ye naturally aata hai jab koi cheez continuously grow karti hai.
Example: Bacteria har second double hote hain → growth ka pattern e se follow hota hai.

Key rules (bas yehi chahiye):
  e^0 = 1          (koi bhi cheez ki power 0 = 1)
  e^(bada +) = bahut bada number
  e^(bada -) = bahut chhota number (almost 0)
```

### Ab Sigmoid Formula

```
                   1
  σ(z)  =  ──────────────
             1 + e^(-z)
```

**Ye formula bol raha hai:**
- Mujhe koi bhi number `z` do
- Main tumhe 0 se 1 ke beech ek value dunga (probability!)

### Step-by-Step: Formula Kaise Kaam Karta Hai?

**Example 1: z = 0 (bilkul beech mein)**
```
σ(0) = 1 / (1 + e^(-0))
     = 1 / (1 + e^0)
     = 1 / (1 + 1)          ← e^0 = 1
     = 1 / 2
     = 0.5                  ← EXACTLY 50-50! Undecided!
```

**Example 2: z = 10 (bahut bada positive)**
```
σ(10) = 1 / (1 + e^(-10))
      = 1 / (1 + 0.0000454)   ← e^(-10) bahut chhota number hai
      = 1 / 1.0000454
      ≈ 0.99995               ← ALMOST 1! Very confident = YES!
```

**Example 3: z = -10 (bahut bada negative)**
```
σ(-10) = 1 / (1 + e^(10))
       = 1 / (1 + 22026)      ← e^10 bahut bada number hai
       = 1 / 22027
       ≈ 0.000045             ← ALMOST 0! Very confident = NO!
```

### Pattern Dekho:

```
z = -∞  →  σ(z) = 0           (pakka NO)
z = -5  →  σ(z) = 0.007       (lagbhag NO)
z = -2  →  σ(z) = 0.12        (probably NO)
z =  0  →  σ(z) = 0.50        (pata nahi, 50-50)
z = +2  →  σ(z) = 0.88        (probably YES)
z = +5  →  σ(z) = 0.993       (lagbhag YES)
z = +∞  →  σ(z) = 1           (pakka YES)
```

### S-Curve Kaise Dikhta Hai:

```
  σ(z)
  1.0  ─ ─ ─ ─ ─ ─ ─ ─ ── ─ ────────────
                              ╱
  0.8                       ╱
                          ╱
  0.5  ─ ─ ─ ─ ─ ─ ─ ✕ ─ ─ ─ ─ ─ ─ ─ ─    ← z = 0 pe exactly 0.5
                    ╱
  0.2             ╱
                ╱
  0.0  ────────── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
       -10  -5   0   +5  +10
                  z →
```

**Dekh — S shape hai! Smoothly 0 se 1 pe jaata hai. PERFECT for probability!**

### Lekin z Kya Hai? — Ye Wahi Linear Regression Wala Formula Hai!

```
z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ

Jahan:
  w₀ = intercept (bias)
  w₁, w₂, ... = weights (har feature ki importance)
  x₁, x₂, ... = features (input data)
```

**MATLAB — Logistic Regression = Linear Regression + Sigmoid wrapper!**

```
Linear Regression:      answer = w₀ + w₁x₁ + w₂x₂           → koi bhi number
Logistic Regression:    answer = sigmoid(w₀ + w₁x₁ + w₂x₂)  → 0 se 1 (probability!)
```

### Real Example — Cancer Prediction

```
Features:
  x₁ = Tumor Size (cm)
  x₂ = Patient Age (years)

Model ne training ke baad ye weights seekhe:
  w₀ = -8.0  (intercept)
  w₁ = +1.5  (tumor size ka weight)
  w₂ = +0.02 (age ka weight)

Ab ek naya patient aaya: Tumor = 5cm, Age = 60

Step 1: z nikalo
  z = -8.0 + (1.5 × 5) + (0.02 × 60)
  z = -8.0 + 7.5 + 1.2
  z = 0.7

Step 2: Sigmoid lagao
  σ(0.7) = 1 / (1 + e^(-0.7))
         = 1 / (1 + 0.4966)
         = 1 / 1.4966
         = 0.668

Step 3: Interpret karo
  P(Cancer) = 0.668 = 66.8%
  
  Since 66.8% > 50% → Predict: CANCER (Class 1)
```

**Ye hai Logistic Regression — bas itna hi hai! Linear combo → Sigmoid → Probability → Decision!**

---

## PART 3: Decision Boundary — Kahan Cut Lagayen?

### Default Rule

```
Agar σ(z) ≥ 0.5   →   Predict Class 1 (YES)
Agar σ(z) < 0.5   →   Predict Class 0 (NO)

σ(z) = 0.5 tab hota hai jab z = 0

Toh basically:
  z ≥ 0  →  Class 1
  z < 0  →  Class 0
```

### Graph Mein:

```
  Agar 2 features hain (x₁, x₂), toh:
  
  z = w₀ + w₁x₁ + w₂x₂ = 0   ← YE EK LINE HAI x₁-x₂ plane mein!
  
     x₂
      │  Class 1 (YES) ★          Decision boundary
      │  ★   ★   ★               (jahan z = 0)
      │    ★   ★                        │
      │  ★   ╱──────────────────────────
      │     ╱    ○   ○
      │    ╱  ○    ○    ○
      │   ╱    ○   ○
      │  ╱  Class 0 (NO) ○
      └────────────────────── x₁

  Ek side: z > 0 → σ(z) > 0.5 → Class 1
  Doosri side: z < 0 → σ(z) < 0.5 → Class 0
```

**Important point:** Logistic Regression ka decision boundary **HAMESHA ek straight line** hota hai.
(Agar manually x₁², x₁×x₂ jaise features add karo toh curved ban sakta hai — lekin naturally nahi)

### Threshold Change Karna

Default 0.5 hai, lekin hum change kar sakte hain!

**Cancer detection mein:**
```
Default: P ≥ 0.5 → Cancer
Problem: Agar patient ko 45% probability hai cancer ka, hum "No Cancer" bolenge
         lekin agar galat nikla? PATIENT MAR SAKTA HAI!

Better: P ≥ 0.3 → Cancer (safe side pe raho)
         Zyada patients ko "Cancer" bolenge → kuch false alarms,
         lekin koi actual cancer patient MISS nahi hoga!
```

```
Spam detection mein:
  P ≥ 0.7 → Spam (strict raho, important email spam mein na jaaye)

Cancer detection mein:
  P ≥ 0.2 → Cancer (loose raho, koi cancer miss na ho)
```

---

## PART 4: Model Kaise Seekhta Hai? (Training — Simple Version)

### Problem:

Model ke paas initially RANDOM weights hain (w₀, w₁, w₂...)
Isko "sahi" weights dhundhne hain taki predictions correct aayein.

### Loss Function — Kitna Galat Hai Model?

Humein ek number chahiye jo bataye: "Model kitna galat predictions de raha hai?"

**Kyun MSE (Mean Squared Error) yahan nahi chalega:**
```
MSE mein hum karte hain: (actual - predicted)²

Logistic mein predicted = sigmoid(z) → ek curved function hai
MSE + sigmoid = ek aisa graph ban jaata hai jisme kaafi "gaddhe" hain

                                                      
  MSE Loss         Log Loss         
    ╲ ╱╲ ╱             ╲       ╱     
     ╳  ╳               ╲     ╱      
    ╱ ╲╱ ╲               ╲   ╱       
                           ╲ ╱        
  Multiple                 ▼         
  gaddhe!            Ek hi gaddha    
  (STUCK ho           (GUARANTEED    
   jaoge!)              pahucho!)    
```

**Is liye hum LOG LOSS use karte hain:**

### Log Loss — Intuition (Bina Formula Ke)

Soch aise:

```
Actual answer: Cancer HAI (y = 1)

Model bole: "99% chance cancer hai" (p = 0.99)
→ BAHUT ACHHA! → Loss BAHUT KAM

Model bole: "50% chance cancer hai" (p = 0.50)
→ Theek theek... → Loss MEDIUM

Model bole: "1% chance cancer hai" (p = 0.01)
→ BILKUL GALAT! → Loss BAHUT ZYADA
```

```
Actual answer: Cancer NAHI hai (y = 0)

Model bole: "1% chance cancer hai" (p = 0.01)
→ BAHUT ACHHA! → Loss BAHUT KAM

Model bole: "99% chance cancer hai" (p = 0.99)
→ BILKUL GALAT! → Loss BAHUT ZYADA
```

### Log Loss Formula (Simple Version)

```
Ek patient ke liye:

  Agar actual y = 1:   Loss = -log(p)
  Agar actual y = 0:   Loss = -log(1-p)

Combined formula:
  Loss = -[ y × log(p) + (1-y) × log(1-p) ]
```

**Verify karo:**
```
y = 1, p = 0.99:  Loss = -log(0.99)  = 0.01   ← bahut kam, sahi prediction!
y = 1, p = 0.50:  Loss = -log(0.50)  = 0.69   ← medium
y = 1, p = 0.01:  Loss = -log(0.01)  = 4.60   ← bahut zyada, galat prediction!
y = 0, p = 0.01:  Loss = -log(0.99)  = 0.01   ← bahut kam, sahi prediction!
y = 0, p = 0.99:  Loss = -log(0.01)  = 4.60   ← bahut zyada, galat prediction!
```

**Log kya karta hai yahan?**
```
log ek aisa function hai jo:
  - log(1) = 0          ← perfect prediction pe loss = 0
  - log(0.5) = -0.69    ← uncertain pe medium loss
  - log(0.01) = -4.60   ← galat pe BAHUT bada negative

Uske aage minus (-) laga diya toh positive loss mil gaya!
  -log(0.99) = 0.01    (sahi = low loss)
  -log(0.01) = 4.60    (galat = high loss)
```

### Training Process (Step by Step)

```
1. Random weights se start karo (w₀ = 0, w₁ = 0, w₂ = 0)
2. Har patient ke liye predict karo: z = w₀ + w₁x₁ + w₂x₂ → p = σ(z)
3. Log Loss calculate karo (kitna galat hai)
4. Weights ko THODA adjust karo taaki loss KAM ho
5. Step 2-4 repeat karo bahut baar (1000 times!)
6. Eventually weights "best" values pe pahunch jaate hain

Ye process hai GRADIENT DESCENT — same jo Linear Regression mein tha!
```

---

## PART 5: Regularization — Overfitting Se Bachao

### Problem

Agar model bahut complex weights seekh le → training data pe perfect, naye data pe fail!

Regression mein bhi tha ye problem:
```
Ridge Regression: weights ko chhota rakho (L2 penalty)
Lasso Regression: kuch weights zero kar do (L1 penalty)
```

### Logistic Regression Mein: C Parameter

```
LogisticRegression(C=1.0)

C = Regularization ka INVERSE!

  C bahut CHHOTA (0.001):
    → Strong regularization
    → Weights bahut chhote rahenge
    → Simple model → might UNDERFIT
    
  C bahut BADA (1000):
    → Weak regularization
    → Weights bade ho sakte hain
    → Complex model → might OVERFIT
    
  C = 1 (Default):
    → Balanced — usually achha kaam karta hai
```

**YAAD RAKH — ye Ridge/Lasso se ULTA hai:**
```
Ridge/Lasso:  alpha BADHAO = zyada regularization
LogisticReg:  C BADHAO     = KAM regularization   (kyunki C = 1/alpha)
```

### Penalty Types

```python
LogisticRegression(penalty='l2')   # Default — Ridge jaisa (weights chhote)
LogisticRegression(penalty='l1')   # Lasso jaisa (kuch weights zero)
```

---

## PART 6: Multiclass — Agar 2 Se Zyada Classes Hain?

Logistic Regression default mein 2 classes ke liye hai (Yes/No).

Lekin kya karein agar 3 classes hain? Jaise:
```
Iris flower → Setosa / Versicolor / Virginica
```

### Method: One-vs-Rest (OvR)

```
3 classes ke liye 3 ALAG binary classifiers banao:

Model 1: Setosa (YES) vs Baaki (NO)        → P(Setosa) = 0.85
Model 2: Versicolor (YES) vs Baaki (NO)    → P(Versicolor) = 0.10
Model 3: Virginica (YES) vs Baaki (NO)     → P(Virginica) = 0.05

Final answer: Jis class ka P sabse bada → wo predict!
→ Predict: SETOSA (85% sure)
```

sklearn automatically handle karta hai ye — tujhe kuch manually nahi karna.

---

## PART 7: predict() vs predict_proba() — Bahut Important!

```python
model.predict(X_test)
# Output: [1, 0, 1, 1, 0]
# Sirf labels — "Cancer" ya "No Cancer"

model.predict_proba(X_test)
# Output: [[0.15, 0.85],    ← 85% cancer
#          [0.92, 0.08],    ← 8% cancer
#          [0.03, 0.97],    ← 97% cancer
#          [0.45, 0.55],    ← 55% cancer  (barely!)
#          [0.88, 0.12]]   ← 12% cancer
```

**predict_proba() ZYADA useful hai kyunki:**

```
Patient A: predict = 1, predict_proba = 97% cancer
→ Doctor ko bolo: "Almost confirm cancer hai, immediately treat karo"

Patient B: predict = 1, predict_proba = 51% cancer  
→ Doctor ko bolo: "Borderline hai, ek aur test kara lo"

predict() toh dono ko same "1" bolega — ye difference kho jaata hai!
```

---

## PART 8: Coefficients Ka Matlab — Model Kya Seekha?

Training ke baad model ke paas weights (coefficients) hote hain.

```python
model.coef_        # [0.5, -0.3, 1.2]  (har feature ka weight)
model.intercept_   # [-2.1]             (bias/intercept)
```

### Interpretation

```
Agar feature "Tumor Size" ka coefficient = +1.5

Matlab:
  Tumor Size 1 unit badhne pe → z mein +1.5 ka increase
  → probability BADHEGI (towards cancer)
  
Agar feature "Exercise Hours" ka coefficient = -0.8

Matlab:
  Exercise 1 unit badhne pe → z mein -0.8 ka decrease
  → probability GHATTEGI (away from cancer)

Rules:
  POSITIVE coefficient → feature badhne se Class 1 ki probability BADHTI hai
  NEGATIVE coefficient → feature badhne se Class 1 ki probability GHATTI hai
  ZERO coefficient     → feature ka koi asar hi nahi
  
  |coefficient| BADA → feature ZYADA important
```

---

## PART 9: Common Mistakes — In Se Bachna!

### Mistake 1: Scaling Bhool Jaana

```
Feature 1: Age          → 20 to 80       (range = 60)
Feature 2: Salary       → 20000 to 500000 (range = 480000)

Bina scaling ke Salary ka coefficient CHHOTTHA dikhega (kyunki values badi hain)
lekin importance KAM nahi hai!

FIX: Hamesha StandardScaler use karo pipeline mein!
```

### Mistake 2: C ko alpha samajhna

```
Ridge: alpha = 10  → STRONG regularization
LogReg: C = 10     → WEAK regularization!

YE ULTA HAI! C = 1/alpha
```

### Mistake 3: max_iter kam rakhna

```python
# Ye error aayega:
# "ConvergenceWarning: lbfgs failed to converge"

# FIX:
LogisticRegression(max_iter=10000)  # Default 100 chhota hai, badhao
```

---

## PART 10: Full Example — Step by Step (Cancer Prediction)

### Step 1: Data Load Karo

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data      # 569 patients, 30 features (tumor measurements)
y = data.target    # 0 = Malignant (cancer), 1 = Benign (no cancer)
```

### Step 2: Split Karo

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 80% training, 20% testing
# stratify=y → dono sets mein cancer/benign ka ratio same rahega
```

### Step 3: Pipeline Banao (Scaler + Model)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),           # Step 1: Scale features
    ('model', LogisticRegression(           # Step 2: Train model
        C=1.0,                              # Default regularization
        max_iter=10000,                     # Enough iterations
        random_state=42
    ))
])
```

### Step 4: Train Karo

```python
pipe.fit(X_train, y_train)
# Ye andar hi andar:
# 1. Scaler ne X_train ke mean/std seekhe
# 2. X_train ko scale kiya
# 3. LogisticRegression ne scaled data pe weights seekhe
```

### Step 5: Evaluate Karo

```python
train_acc = pipe.score(X_train, y_train)
test_acc = pipe.score(X_test, y_test)

print(f"Train Accuracy: {train_acc:.4f}")   # ~0.989
print(f"Test Accuracy:  {test_acc:.4f}")     # ~0.982
# 98.2% accuracy — bahut achha!
```

### Step 6: Predictions Dekho

```python
y_pred = pipe.predict(X_test)        # [1, 0, 1, 1, ...] labels
y_proba = pipe.predict_proba(X_test) # [[0.02, 0.98], ...] probabilities

# Ek patient dekho:
print(f"Prediction: {y_pred[0]}")          # 1 (Benign)
print(f"Confidence: {y_proba[0][1]:.1%}")  # 98.3% sure
```

### Step 7: Coefficients Dekho (Kaunse Features Important Hain?)

```python
model = pipe.named_steps['model']
import numpy as np

# Top 5 most important features:
coefs = model.coef_[0]
top_5 = np.argsort(np.abs(coefs))[::-1][:5]

for idx in top_5:
    print(f"  {data.feature_names[idx]:30s} | coef = {coefs[idx]:+.4f}")

# Positive coef = zyada hone pe cancer KAM probable (Benign)
# Negative coef = zyada hone pe cancer ZYADA probable (Malignant)
```

### Summary — Poora Flow Ek Nazar Mein

```
┌─────────────────────────────────────────────────────────┐
│                  LOGISTIC REGRESSION FLOW                │
│                                                          │
│  Data → Split → Scale → Train → Predict → Evaluate      │
│                                                          │
│  Internally:                                             │
│  x₁, x₂, ... → z = w₀ + w₁x₁ + w₂x₂ → p = σ(z) → 0/1│
│                    ↑                        ↑            │
│              Linear combo             Sigmoid squishes   │
│              (same as LinReg)         into [0, 1]        │
│                                                          │
│  Loss: Log Loss (not MSE!)                               │
│  Regularization: C parameter (C↑ = less reg)             │
│  Multiclass: OvR (automatic in sklearn)                  │
│  ALWAYS SCALE YOUR DATA!                                 │
└─────────────────────────────────────────────────────────┘
```

---

## Practice Questions

**Q1.** σ(0) kitna hoga? Manually calculate karo. Hint: e^0 = 1

**Q2.** Agar kisi patient ka z = -3 aaye, toh:
  - σ(z) kitna hoga approximately? 
  - Model kya predict karega?
  - Kya tum confident ho is prediction mein?

**Q3.** C = 0.001 vs C = 100 — dono mein kya farak aayega model mein? Kaunsa overfit karega?

**Q4.** Agar model predict_proba de: [0.48, 0.52] — kya tum is prediction pe bharosa karoge? Kyun?

**Q5.** Ek dataset mein 2 features hain. Model ne seekha: w₀ = -1, w₁ = 2, w₂ = -3.
  - Agar x₁ = 2, x₂ = 1 → Predict karo (pura process: z → σ(z) → class)
  - Kaunsa feature zyada important hai?

---

> **Next: Classification Metrics (Ch 2) — Model achha hai ya bura, kaise measure karein?**
> **Code file: `code/classi/ch1_logistic_regression.py`**
