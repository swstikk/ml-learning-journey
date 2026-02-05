# 🚀 ML Learning Journey

> My journey from ML basics to advanced - documenting everything I learn!
> Started: February 2026 | Age: 16

---

## 📊 Current Progress

```
[■■■■■■■□□□] 70% Regression Complete
```

---

## 📚 Chapters Completed

### ✅ Chapter 1: Simple Linear Regression
- OLS Formula derivation
- Manual m and b calculation
- R² and RMSE understanding

### ✅ Chapter 2: Multiple Linear Regression
- Design Matrix concept
- Matrix OLS: β = (XᵀX)⁻¹Xᵀy
- Coefficient interpretation

### ✅ Chapter 3: Regression Diagnostics
- Residuals vs Fitted plots
- Durbin-Watson test
- VIF for multicollinearity

### ✅ Chapter 4: Polynomial Regression
- When linear doesn't fit
- PolynomialFeatures usage
- Overfitting concepts

### ✅ Chapter 5: Regularization
- Ridge (L2)
- Lasso (L1)
- ElasticNet

### 🔄 In Progress: Evaluation Metrics

---

## 📁 Repository Structure

```
ml-learning-journey/
├── README.md
├── quizzes/
│   ├── chapter1_3_quiz_linear_regression.py   # Ch 1-3 practice
│   └── chapter4_quiz_polynomial_regression.py # Ch 4 practice
└── code/
    └── (more files coming soon)
```

---

## 🎯 Learning Path

Following a structured curriculum:
1. ~~Simple Regression~~ ✅
2. ~~Multiple Regression~~ ✅
3. ~~Diagnostics~~ ✅
4. ~~Polynomial Regression~~ ✅
5. ~~Regularization~~ ✅
6. Evaluation Metrics 🔄
7. Cross-Validation
8. Projects!

---

## 📝 Quiz Scores

| Chapter | Topic | Score |
|---------|-------|-------|
| 1-3 | Linear Regression | ✅ |
| 4 | Polynomial Regression | 32/35 |

---

## 💡 Key Learnings

### The Matrix OLS Formula
```
β = (XᵀX)⁻¹Xᵀy

Where:
- X = Design Matrix (features with intercept column)
- y = Target values
- β = Coefficients [b, m1, m2, ...]
```

### fit() vs fit_transform() vs transform()
```python
# Training data: Learn + Apply
X_train_transformed = poly.fit_transform(X_train)

# Test data: ONLY Apply (never fit on test!)
X_test_transformed = poly.transform(X_test)
```

---

## 🔗 Connect

- GitHub: [@swstikk](https://github.com/swstikk)

---

*Learning in public, one commit at a time! 🧠*
