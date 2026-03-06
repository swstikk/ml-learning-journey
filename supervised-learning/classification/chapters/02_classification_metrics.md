# 📔 Classification Chapter 2: Classification Metrics — Complete Guide

> **"Accuracy is a LIAR. Learn the real metrics."**

---

## 1. The Problem — Why Accuracy Alone Fails 🚨

Imagine a hospital cancer detection system:
- 1000 patients, but only **10** actually have cancer (1%)
- A dumb model that predicts **"No Cancer" for everyone** → **99% Accuracy!** 🎉
- But it missed ALL 10 cancer patients!! → USELESS.

```
Population: 1000 patients
  Cancer:        10  (1%)
  No Cancer:    990  (99%)

Dumb Model: "Predict No Cancer for everyone"
  Accuracy = 990/1000 = 99% ← MISLEADING!
  Cancer patients detected = 0/10 = 0% ← DISASTER!
```

**Yahi problem hai accuracy ki** — when data is **imbalanced**, accuracy is misleading.
Isi liye humein aur metrics chahiye.

---

## 2. Confusion Matrix — The Foundation of Everything 🧱

Har prediction 4 categories mein girti hai:

```
                         PREDICTED
                    Negative (0)    Positive (1)
                 ┌────────────────┬────────────────┐
  ACTUAL    0    │  TN (True Neg) │  FP (False Pos)│  ← Type I Error
  (Truth)        │  Sahi reject   │  False alarm!  │
                 ├────────────────┼────────────────┤
            1    │  FN (False Neg)│  TP (True Pos) │  ← Type II Error
                 │  MISSED!       │  Correct catch │
                 └────────────────┴────────────────┘
```

### Real-World Analogy — Fire Alarm:
- **TP**: Alarm bajai + fire tha → Correct!
- **TN**: Alarm nahi bajai + fire nahi tha → Correct!  
- **FP**: Alarm bajai + fire nahi tha → **False Alarm** (Type I — annoying)
- **FN**: Alarm nahi bajai + fire tha → **MISSED FIRE!** (Type II — DANGEROUS)

### Key Rule:
```
"False" means the model was WRONG.
"Positive/Negative" means what the model PREDICTED.

FP = Model predicted Positive, but it was False (wrong) → Actually Negative.
FN = Model predicted Negative, but it was False (wrong) → Actually Positive.
```

---

## 3. Accuracy — The Basics (and Its Limits)

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN} = \frac{Correct}{Total}$$

### When Accuracy is OK:
- Balanced datasets (50-50 split, ya close)
- When FP and FN equally costly

### When Accuracy LIES:
- **Imbalanced data** (99% one class → dumb model = 99% accuracy)
- When cost of FP ≠ cost of FN (cancer detection, fraud detection)

---

## 4. Precision — "Jab Model Positive Bole, Kitni Baar Sahi?"

$$Precision = \frac{TP}{TP + FP}$$

**English:** Of all the times model said "Positive", how many were actually positive?

### Real-World:
- **Spam Filter**: Precision = "Of emails marked as spam, how many were actually spam?"
  - Low precision → good emails going to spam folder → ANNOYING
- **Stock Buy Signal**: "Jab bola buy, kitni baar actually profit hua?"

### When to Prioritize Precision:
- When **False Positive is costly**
- Spam filter (don't want real email in spam)
- Recommender system (don't recommend wrong items)
- Legal system (don't convict innocent person)

---

## 5. Recall (Sensitivity / TPR) — "Actual Positives Mein Se Kitne Pakde?"

$$Recall = \frac{TP}{TP + FN}$$

**English:** Of all actual positives, how many did the model catch?

### Real-World:
- **Cancer Detection**: Recall = "Of patients who had cancer, how many did we detect?"
  - Low recall → missed cancer patients → DEATH
- **Fraud Detection**: "Actual fraud cases mein se kitne detect kiye?"

### When to Prioritize Recall:
- When **False Negative is costly**
- Cancer screening (don't miss cancer!)
- Fraud detection (don't miss fraud!)
- Security threats (don't miss an intruder!)

---

## 6. Precision vs Recall — The Tradeoff ⚖️

```
  Precision HIGH + Recall LOW:
    - Model bohot choosy hai
    - "Jab bola positive, toh sahi tha, but bahut saare miss bhi kiye"
    - Example: Only predict cancer if 99% sure → miss mild cases

  Precision LOW + Recall HIGH:
    - Model sab ko positive bol raha
    - "Sab catch kar liye, but bahut saare false alarms bhi"
    - Example: Everyone gets flagged as cancer → nobody missed, but 900 false positives

  TRADEOFF: Ek badhao toh doosra ghatta hai!
```

```
         High Precision                    High Recall
         ┌───────────┐                    ┌───────────┐
         │  Careful   │                    │  Catches  │
         │  but misses│   ← TRADEOFF →    │  all but  │
         │  some      │                    │  noisy    │
         └───────────┘                    └───────────┘
```

---

## 7. F1 Score — The Harmonic Mean Balance

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

- F1 is the **harmonic mean** of Precision and Recall
- Harmonic mean penalizes extreme differences (unlike arithmetic mean)
- F1 = 1.0 → Perfect | F1 = 0.0 → Worst

### Why Harmonic Mean, Not Arithmetic?
```
  Precision = 1.0, Recall = 0.0
  Arithmetic Mean = (1.0 + 0.0) / 2 = 0.5  ← Seems OK? But model is USELESS
  Harmonic Mean   = 2 * (1*0)/(1+0) = 0.0  ← Correctly says: USELESS!

  Precision = 0.9, Recall = 0.1
  Arithmetic Mean = 0.50  ← Looks decent
  Harmonic Mean   = 0.18  ← Correctly low!
```

### When to Use F1:
- When you need balance between Precision and Recall
- Imbalanced datasets (F1 > Accuracy)
- When both FP and FN matter

---

## 8. Specificity, FPR, and the ROC Connection

$$Specificity = \frac{TN}{TN + FP}$$

**English:** Of actual negatives, how many correctly identified as negative?
(Basically Recall for the negative class)

$$FPR = 1 - Specificity = \frac{FP}{TN + FP}$$

**English:** Of actual negatives, how many falsely called positive?

These are needed for ROC curves.

---

## 9. ROC Curve & AUC — The Gold Standard 🏆

### What is ROC?
ROC = **Receiver Operating Characteristic**
It plots **TPR (Recall) vs FPR** at different classification thresholds.

```
  Threshold = 0.9 → Very few positives → High Precision, Low Recall
  Threshold = 0.5 → Default balance
  Threshold = 0.1 → Almost everyone positive → Low Precision, High Recall

  ROC curve traces how TPR and FPR change as threshold moves from 1.0 to 0.0
```

### AUC (Area Under ROC Curve):
- **AUC = 1.0** → Perfect classifier
- **AUC = 0.5** → Random (diagonal line = coin flip)
- **AUC < 0.5** → Worse than random (something wrong!)
- **AUC > 0.9** → Excellent
- **AUC 0.8-0.9** → Good
- **AUC 0.7-0.8** → Fair
- **AUC < 0.7** → Poor

### Why AUC is Great:
- **Threshold independent** — measures overall model quality
- **Works on imbalanced data** — unlike accuracy
- **Comparable across models** — higher AUC = better model

---

## 10. Precision-Recall Curve — Better for Imbalanced Data

ROC curve can be misleading on highly imbalanced data (AUC can look good even with bad model).

**PR Curve** plots Precision vs Recall at different thresholds.
**AP (Average Precision)** = Area under PR curve

### When to Use PR Curve vs ROC:
```
  Balanced data       → ROC + AUC
  Imbalanced data     → PR Curve + AP
  High negative class → PR Curve (more informative)
```

---

## 11. Threshold Tuning — 0.5 is NOT Always Best 🎯

Default threshold = 0.5. But you can MOVE it!

```
  Cancer detection:
    Lower threshold (0.3) → More people flagged → Higher Recall
    "Better to have false alarms than miss cancer"

  Spam filter:
    Higher threshold (0.8) → Only confident spam filtered → Higher Precision
    "Better to let some spam through than lose real email"
```

### How to Find Optimal Threshold:
1. Plot Precision, Recall, F1 at various thresholds (0.1 to 0.9)
2. Pick the threshold that maximizes your chosen metric
3. Or use the Precision-Recall curve's "elbow"

---

## 12. Multi-Class Metrics — Macro vs Weighted vs Micro

When you have 3+ classes (e.g., Iris: Setosa, Versicolor, Virginica):

### Macro Average:
- Calculate metric for EACH class → Take simple average
- **Treats all classes equally** (even rare ones)
```
  F1_setosa = 0.95, F1_versicolor = 0.80, F1_virginica = 0.85
  Macro F1 = (0.95 + 0.80 + 0.85) / 3 = 0.867
```

### Weighted Average:
- Calculate metric for each class → Weighted average by class size
- **Accounts for class imbalance**

### Micro Average:
- Aggregate all TP, FP, FN globally → Calculate single metric
- Equivalent to accuracy in multi-class

### When to Use Which:
```
  All classes equally important  → Macro
  Want to account for class sizes → Weighted
  Overall performance number    → Micro
```

---

## 13. Class Imbalance — The Real-World Problem 🏥

Most real datasets are imbalanced: Fraud (0.1%), Cancer (1%), Defects (5%).

### Solutions:

**A. class_weight='balanced' (sklearn)**
- Automatically adjusts weights → rare class gets more importance
- `LogisticRegression(class_weight='balanced')`

**B. Oversampling (SMOTE)**
- Create synthetic samples of minority class
- `from imblearn.over_sampling import SMOTE`

**C. Undersampling**
- Remove samples from majority class
- Simple but loses information

**D. Threshold adjustment**
- Lower threshold for rare class

```
  Problem: 1000 samples, 990 negative, 10 positive
  
  Without handling → Model predicts all negative → 99% accuracy, 0% recall
  
  With class_weight='balanced':
    Weight_positive = 1000 / (2 * 10) = 50
    Weight_negative = 1000 / (2 * 990) = 0.505
    Model pays 50x more attention to positive class!
```

---

## 14. The Decision Matrix — Which Metric When? 🗺️

| Situation | Primary Metric | Why |
|-----------|---------------|-----|
| Balanced data, general | **Accuracy** | Simple, meaningful |
| Imbalanced data | **F1 Score** | Balance P & R |
| FP is costly (spam) | **Precision** | Don't false alarm |
| FN is costly (cancer) | **Recall** | Don't miss positives |
| Overall model quality | **AUC-ROC** | Threshold-independent |
| Very imbalanced | **PR-AUC** | More honest than ROC |
| Business cost matrix | **Custom metric** | Weight by actual costs |

---

## 🎯 Practice Questions (10)

### Theory:
1. Ek model ka Precision = 0.95, Recall = 0.10 hai. Iska matlab kya hai real-world mein?  
2. ROC-AUC = 0.50 ka matlab kya hai? Yeh kab hota hai?
3. Fraud detection (0.1% fraud) mein Accuracy kyun misleading hai?
4. F1 Score harmonic mean kyun use karta hai, arithmetic mean kyun nahi?
5. Threshold 0.5 se 0.3 karne se Precision pe kya effect hoga? Recall pe?

### Coding:
6. Breast Cancer dataset pe manually TP, TN, FP, FN calculate karo (without sklearn).
7. From-scratch Precision, Recall, F1 function banao aur sklearn se compare karo.
8. ROC curve plot karo aur AUC calculate karo. Model compare karo (LogReg vs DummyClassifier).
9. Imbalanced dataset banao (95-5 split) aur dikhao ki accuracy misleading hai lekin F1 sahi batata hai.
10. Threshold tuning: Plot Precision, Recall, F1 vs threshold aur optimal threshold dhundho.

---

> **Next: Chapter 3 — Decision Trees** 🌳
