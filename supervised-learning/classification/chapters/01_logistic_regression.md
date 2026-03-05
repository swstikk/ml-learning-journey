# 📔 ML Module 21: Logistic Regression
> **"It's not regression, it's classification."**

---

## 1. The Big Idea
In Linear Regression, we predicted a **continuous number** (e.g., Salary = ₹55,032).
In Logistic Regression, we predict a **class** or **probability** (e.g., Has Cancer = Yes/No, or 88% chance).

Despite the name, Logistic Regression is used for **Classification**.

### Why not just use Linear Regression?
If you try to fit a straight line to data points that are either 0 or 1:
1.  **Outliers break it:** A single outlier can shift the line so much that it predicts probabilities > 1 or < 0.
2.  **Meaningless predictions:** What does a prediction of 1.5 mean for a Yes/No question?
3.  **Non-Linearity:** Binary data doesn't follow a straight line; it follows an **S-curve**.

---

## 2. The Sigmoid Function (The Secret Sauce) 🧪
To turn our linear equation into a probability, we wrap it in the **Sigmoid (Logistic) Function**.

**Linear Equation:** $z = w_0 + w_1x_1 + w_2x_2 + ...$
**Sigmoid Wrap:** $P = \frac{1}{1 + e^{-z}}$

### Characteristics of Sigmoid:
- If $z$ is a very large positive number, $P \approx 1$.
- If $z$ is a very large negative number, $P \approx 0$.
- If $z = 0$, $P = 0.5$ (The turning point).

---

## 3. Decision Boundary
Once we have the probability $P$:
- If $P \geq 0.5 \rightarrow$ **Class 1** (Success/Yes)
- If $P < 0.5 \rightarrow$ **Class 0** (Failure/No)

The line where $P = 0.5$ (or $z = 0$) is called the **Decision Boundary**. It separates the two classes.

---

## 4. The Loss Function: Log Loss (Cross-Entropy) 📉
We can't use Mean Squared Error (MSE) here because the Sigmoid function makes the MSE "non-convex" (it has many local minima, so Gradient Descent gets stuck).

Instead, we use **Log Loss**:
- If the true label is $y=1$ and we predict $P=0.999$, loss is near 0.
- If the true label is $y=1$ and we predict $P=0.001$, loss is very high (infinity).

**Formula for one point:**
$Loss = -[y \cdot \log(p) + (1-y) \cdot \log(1-p)]$

---

## 5. Hyperparameters
- **C:** The inverse of regularization strength ($C = 1/\lambda$).
  - **Small C:** Strong regularization (prevents overfitting, smaller weights).
  - **Large C:** Weak regularization (tries to fit training data perfectly).
- **Penalty:** 'l1' (Lasso) or 'l2' (Ridge).

---

## 6. Implementation Checklist
1.  [ ] Import `LogisticRegression` from `sklearn.linear_model`.
2.  [ ] **Scale your data!** Logistic Regression is sensitive to feature scale.
3.  [ ] Fit the model.
4.  [ ] Use `predict()` for labels (0/1).
5.  [ ] Use `predict_proba()` for probabilities (e.g., [0.2, 0.8]).

---

## 🎯 Quick Quiz
1.  If $z = 5$ in the sigmoid formula, will the predicted class be 0 or 1?
2.  What happens to the model if we set $C = 0.00001$?
3.  True/False: Logistic Regression can only be used for binary classification.

---

> **Ready to code? Check `code/10_logistic_regression.py` next.**
