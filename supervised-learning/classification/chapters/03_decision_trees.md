# Chapter 3: Decision Trees — YES/NO Questions Ka Tree

> **Sabse intuitive model — insaan jaisa sochta hai!**
> **Math level: Bas fractions aur percentages.**

---

## PART 1: Kya Hai Decision Tree?

### Real Life Mein Decision Tree

Tu daily decision trees use karta hai:

```
"Kya bahar baarish ho rahi hai?"
      │
   ┌──┴──┐
  YES     NO
   │       │
"Umbrella  "Kya dhoop hai?"
 le ja"         │
             ┌──┴──┐
            YES     NO
             │       │
         "Sunscreen  "Chal nikal
          laga le"    seedha"
```

**Decision Tree = Questions ka ek tree jisme har node pe ek YES/NO question hai.**
**Neeche jaate jaate final answer (prediction) tak pahunchte ho.**

### ML Mein:

```
"Kya Tumor Size > 4 cm?"
         │
     ┌───┴───┐
    YES       NO
     │         │
"Kya Age > 50?" "BENIGN" ← Leaf node (final answer)
     │
  ┌──┴──┐
 YES     NO
  │       │
"CANCER" "BENIGN"
```

Model khud seekhta hai:
- **Kaunsa question poochna hai?** (Feature select karna)
- **Kya threshold rakhna hai?** (4 cm ya 5 cm?)
- **Kitne questions poochne hain?** (Tree ki depth)

---

## PART 2: Tree Kaise Banta Hai? — Splitting ka Logic

### Problem: Pehla Question Kaunsa Poochein?

10 patients hain:
```
Patient | Tumor Size | Age | Smoker | Cancer?
--------|-----------|-----|--------|--------
   1    |    2 cm   |  30 |   No   |   No
   2    |    3 cm   |  25 |   Yes  |   No
   3    |    5 cm   |  60 |   No   |   Yes
   4    |    1 cm   |  40 |   No   |   No
   5    |    6 cm   |  55 |   Yes  |   Yes
   6    |    4 cm   |  35 |   No   |   No
   7    |    7 cm   |  65 |   Yes  |   Yes
   8    |    8 cm   |  70 |   No   |   Yes
   9    |    2 cm   |  45 |   Yes  |   No
  10    |    5 cm   |  50 |   No   |   Yes
```

Kaunsa question sabse USEFUL hai?
- "Kya Tumor Size > 4.5?" → Left: {1,2,4,6,9}=No, Right: {3,5,7,8,10}=Yes → PERFECT split!
- "Kya Smoker hai?" → Left: {1,3,4,6,8,10}, Right: {2,5,7,9} → Mixed! Not great.

**Best split = jo data ko sabse CLEANLY do groups mein baante!**

### "Clean" Split Kaise Measure Karein? → GINI IMPURITY

**Gini Impurity batata hai: "Agar is group mein se randomly ek sample uthao, toh kitna chance hai ki GALAT classify hoga?"**

```
Gini = 1 - (p₁² + p₂²)

Jahan:
  p₁ = fraction of class 1 (e.g., cancer patients)
  p₂ = fraction of class 0 (e.g., healthy patients)
```

### Examples:

```
Group A: 5 Cancer, 0 Healthy  →  PURE!
  p_cancer = 5/5 = 1.0
  p_healthy = 0/5 = 0.0
  Gini = 1 - (1.0² + 0.0²) = 1 - 1 = 0.0  ← PERFECT (pure)

Group B: 5 Cancer, 5 Healthy  →  50-50 MIXED!
  p_cancer = 5/10 = 0.5
  p_healthy = 5/10 = 0.5
  Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.5  ← WORST (maximum impurity)

Group C: 8 Cancer, 2 Healthy  →  Mostly cancer
  p_cancer = 8/10 = 0.8
  p_healthy = 2/10 = 0.2
  Gini = 1 - (0.8² + 0.2²) = 1 - (0.64 + 0.04) = 0.32  ← Pretty good
```

```
Gini Scale:
  0.0 ──────────── 0.25 ──────────── 0.5
  PURE              OK               MOST MIXED
  (one class only)                   (50-50 split)
```

### Best Split:

```
Algorithm tries EVERY possible split:
  "Tumor > 1?" → calculate Gini of left + right groups
  "Tumor > 2?" → calculate Gini of left + right groups
  "Tumor > 3?" → calculate Gini of left + right groups
  ...
  "Age > 25?" → calculate Gini
  "Age > 30?" → calculate Gini
  ...

Best split = LOWEST weighted Gini of children!

Weighted Gini = (n_left/n_total) × Gini_left + (n_right/n_total) × Gini_right
```

### Alternative: ENTROPY (Information Theory)

```
Entropy = -Σ pᵢ × log₂(pᵢ)

Pure group:   Entropy = 0
50-50 mixed:  Entropy = 1
```

**Gini aur Entropy dono almost same result dete hain. sklearn default Gini use karta hai.**

**Tu abhi itna yaad rakh:**
```
Low Gini  = Clean group = Pure = GOOD
High Gini = Mixed group = Impure = BAD
Tree har step pe LOWEST Gini wala split choose karta hai.
```

---

## PART 3: Overfitting — Decision Trees Ka Sabse Bada Problem!

### Kya Hota Hai Agar Tree Bahut DEEP Ho?

```
Depth = 2 (simple):
  "Tumor > 4.5?"         → Clean, general rules
      ├── YES → Cancer
      └── NO  → Healthy
  
  Train accuracy: 85%
  Test accuracy:  83%    ← GOOD! Dono close!

Depth = 20 (bahut deep):
  "Tumor > 4.5?"
      ├── "Age > 52.3?"
      │     ├── "Smoker AND age > 53.7 AND tumor between 5.1-5.3?"
      │     │     ├── ...
      │     │     └── ...  ← Bahut specific rules!
      └── ...
  
  Train accuracy: 100%   ← Ratta maar liya training data!
  Test accuracy:  65%     ← Naye data pe FAIL!
```

**Ye OVERFITTING hai — model ne training data ka "ratta" maar liya!**

### Solutions — Hyperparameters:

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=5,           # Tree kitna deep ja sakta hai (default: unlimited!)
    min_samples_split=10,  # Split karne ke liye minimum 10 samples chahiye
    min_samples_leaf=5,    # Har leaf mein minimum 5 samples rahein
    max_features='sqrt',   # Har split pe sirf root(n) features consider karo
)
```

```
max_depth = 2:   Too simple → UNDERFIT
max_depth = 100: Too complex → OVERFIT
max_depth = 5-10: Usually sweet spot

Tune kaise? → CROSS VALIDATION!
```

---

## PART 4: Decision Trees Ki Superpowers Aur Weaknesses

### Superpowers:

```
1. NO SCALING NEEDED! 
   → Tree sirf "greater than / less than" dekhta hai
   → Values ka scale matter nahi karta (unlike LogReg, SVM, KNN)

2. INTERPRETABLE!
   → Plot karke dikhao → doctor samajh sakta hai
   → "Kyun ye predict kiya?" ka answer directly milta hai

3. Handles Mixed Data
   → Numbers aur categories dono handle karta hai

4. Feature Importance
   → model.feature_importances_ se pata chalta hai kaunsa feature important hai

5. Non-Linear Decision Boundaries
   → Straight lines nahi — complex shapes bana sakta hai!
```

### Weaknesses:

```
1. OVERFITS EASILY!
   → Bina depth limit ke 100% train accuracy, poor test accuracy

2. UNSTABLE
   → Data mein thoda change → POORA tree badal jaata hai
   → (Random Forest isse fix karta hai — Ch4!)

3. Biased towards features with more values
   → Feature with 100 values vs 2 values → 100 wali ko prefer karega
```

---

## PART 5: Feature Importance — Kaunsa Feature Kitna Important?

```python
model.fit(X_train, y_train)
importances = model.feature_importances_  # Sum = 1.0

# Example output:
# Tumor Size:     0.65  ← 65% important!
# Age:            0.20  ← 20%
# Smoker:         0.10  ← 10%
# Blood Pressure: 0.05  ← 5%
```

**Ye batata hai: "Is feature ne kitna Gini reduce kiya overall tree mein?"**
Jo feature zyada Gini reduce karta hai = zyada important.

### Gene Expression Mein:
```
30,000 genes hain → kaunse genes sabse zyada disease predict karte hain?
Decision Tree ki feature_importances_ → top 10 genes nikal do!
Ye biomarker discovery hai!
```

---

## PART 6: sklearn Code

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

# Train
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

print(f"Train Accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test Accuracy:  {model.score(X_test, y_test):.4f}")

# Visualize Tree!
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=feature_names, 
          class_names=target_names, filled=True, rounded=True)
plt.savefig('decision_tree.png')

# Feature Importance
for name, imp in sorted(zip(feature_names, model.feature_importances_), 
                         key=lambda x: -x[1])[:5]:
    print(f"  {name}: {imp:.4f}")

# Best depth find karo with CV
param_grid = {'max_depth': [2, 3, 5, 7, 10, 15, 20, None]}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                    param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print(f"Best depth: {grid.best_params_['max_depth']}")
```

**NOTICE: No scaler needed! Decision Trees ko scaling ki zarurat NAHI hai!**

---

## Summary

```
Decision Tree = Questions ka tree → split → purer groups → predict
Gini = 1 - sum(pᵢ²) → 0 = pure, 0.5 = most mixed
Overfitting: max_depth control karo!
Superpowers: No scaling, interpretable, feature importance
Weakness: Overfits easily, unstable
Fix: Random Forest (Ch4) = many trees together!
```

## Practice Questions — Theory

**Q1.** 10 samples: 7 Cancer, 3 Healthy. Gini calculate karo.

**Q2.** max_depth=None rakhne pe kya hoga? Ye achha hai ya bura?

**Q3.** Decision Tree ko StandardScaler lagana zaruri hai? Kyun ya kyun nahi?

**Q4.** Tree ki feature_importances_ [0.7, 0.2, 0.1] hai. Kaunsa feature sabse important hai? Ye kaise decide hua?

---

## Coding Challenges — Khud Likho!

Save karo: `code/classi/ch3_practice.py`

### Challenge 1: Gini Impurity From Scratch
```
- Ek function likho: gini_impurity(labels)
    - Input: list of labels, e.g. [0, 0, 1, 1, 1]
    - Output: Gini value
    - Formula: 1 - sum(pᵢ²)
- Test karo:
    gini([1,1,1,1])     = 0.0  (pure)
    gini([1,1,0,0])     = 0.5  (max impurity)
    gini([1,1,1,1,0])   = 0.32
- Print output verify karo
```

### Challenge 2: Overfitting Depth Experiment
```
- Breast Cancer dataset load karo
- max_depth = 1 se 20 tak loop chala
- Har depth pe:
    - DecisionTreeClassifier train karo
    - Train aur Test accuracy store karo
- Plot banao: X = depth, Y = accuracy (2 lines: train, test)
- Mark karo: Best depth kaunsa hai?
- Print: "Overfitting starts at depth = ___"
```

### Challenge 3: Tree Visualization
```
- max_depth=3 ka DecisionTreeClassifier train karo
- plot_tree() se tree visualize karo
- Save as PNG file
- Print karo:
    - Tree ki depth kitni hai?
    - Kitne leaf nodes hain?
    - Root node pe kaunsa feature split hua?
  (Hint: model.tree_.feature[0] se root feature milega)
```

### Challenge 4: Feature Importance Bar Chart
```
- DecisionTreeClassifier train karo (best depth use karo from Challenge 2)
- feature_importances_ nikalo
- Top 10 features ka HORIZONTAL bar chart banao
    - Colors: green for positive, sorted by importance
    - Feature names Y-axis pe
- Print: "Top 3 features: ___"
- Bonus: Kya ye features medically make sense karte hain?
```

### Challenge 5: GridSearchCV — Best Tree
```
- param_grid banao:
    max_depth: [2, 3, 5, 7, 10, 15, None]
    min_samples_split: [2, 5, 10, 20]
    min_samples_leaf: [1, 2, 5, 10]
- GridSearchCV chala (cv=5, scoring='accuracy')
- Print: Best parameters, Best CV score, Test score
- Best model se classification_report print karo
```

---

> **Next: Chapter 4 — Random Forest** (Many trees ka vote = better than one tree!)
