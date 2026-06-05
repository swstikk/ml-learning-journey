# Neural Networks — Ekdum Shuru Se (Pre-Step 0)

> **Ye file 06_neural_networks_deep_math.md se PEHLE padhni hai.**
> **Yahan se tujhe Step 4 (Backprop) tak directly jaana hai.**
> **Koi jump nahi. Koi skip nahi. Har cheez deeply.**

---

## PART A: Ek SINGLE Neuron — Bilkul Andar Se

---

### A1: Data Kaise Aata Hai? Kis Format Mein?

Tere paas ek trade hai. Tujhe predict karna hai: Win hoga ya Loss?

```
Trade #1:
  SL distance  = 0.15
  BB size      = 0.08
```

Computer ko ye dena hai. Computer ko sirf NUMBERS samajh aate hain.

**Input = ek list of numbers:**
```
x = [0.15, 0.08]
```

Bas. Yahi hai "data ka format". Ek list. Ek array. Ek vector.
Isko hum **feature vector** bolte hain.

```
x1 = 0.15   ← pehla feature (SL distance)
x2 = 0.08   ← doosra feature (BB size)
```

Agar tere paas 5 features hote:
```
x = [0.15, 0.08, 25, 1, 14]
     SL    BB    TP  dir  hour

x1=0.15, x2=0.08, x3=25, x4=1, x5=14
```

> **SIMPLE HAI: Input = ek column of numbers. Har number ek feature.**

---

### A2: Weight Kya Hai? — "Kitna Important Hai?"

Ab ek neuron banana hai jo decide kare: Win ya Loss?

**Soch:** Kya SL distance aur BB size dono equally important hain?
Nahi! Ho sakta hai SL distance ZYADA matter kare.

**Isliye har feature ko ek "importance score" dete hain = WEIGHT**

```
w1 = 0.5    ← "SL distance ko 0.5 importance do"
w2 = 0.3    ← "BB size ko 0.3 importance do"
```

**Weights = neuron ka "opinion" ki kaunsi cheez kitni matter karti hai.**

Abhi hume pata nahi SAHI weights kya hain.
Isliye randomly shuru karte hain. Training ke baad sahi ho jayenge.

---

### A3: Weighted Sum — Sab Mix Karo

Ab neuron kya karta hai? Simple:

**Har feature ko uske weight se multiply karo, phir sab jod do:**

```
z = (w1 × x1) + (w2 × x2)
  = (0.5 × 0.15) + (0.3 × 0.08)
  = 0.075 + 0.024
  = 0.099
```

**Ye hai "weighted sum".**

> **Visualize karo:**
> ```
>   x1 = 0.15 ──── ×0.5 ───→ 0.075 ─┐
>                                      ├── ADD ──→ z = 0.099
>   x2 = 0.08 ──── ×0.3 ───→ 0.024 ─┘
> ```

**Soch aise:** Neuron ek "scoring machine" hai.
Wo har feature ko uski importance (weight) se multiply karta hai,
phir sab scores jodta hai = ek final score.

---

### A4: Bias — "Default Mood"

Lekin ek problem hai.

Kya hoga agar sab features 0 hain?
```
z = 0.5 × 0 + 0.3 × 0 = 0
```
z hamesha 0 hoga! Chahe weights kuch bhi hon.

**Hume ek "default value" chahiye jo data se independent ho = BIAS**

```
b = 0.1    ← "default mein thoda positive rakh"

z = (w1 × x1) + (w2 × x2) + b
  = 0.075 + 0.024 + 0.1
  = 0.199
```

> **Bias = neuron ka "default mood"**
> Positive bias: "Mera default hai ki trade jeetega (thoda sa)"
> Negative bias: "Mera default hai ki trade harega"
> Zero bias: "Koi default nahi, sirf data se decide karunga"

**Updated picture:**
```
   x1 = 0.15 ──── ×0.5 ───→ 0.075 ─┐
                                      ├── ADD ──→ z = 0.199
   x2 = 0.08 ──── ×0.3 ───→ 0.024 ─┤
                                      │
   bias = 0.1 ─────────────────────→─┘
```

---

### A5: Pura Formula Ek Line Mein

```
z = w1×x1 + w2×x2 + b

General (n features):
z = w1×x1 + w2×x2 + w3×x3 + ... + wn×xn + b

Short notation:
z = Σ(wi × xi) + b     (i = 1 to n)

Aur bhi short:
z = w⃗ · x⃗ + b          (dot product + bias)
```

**Ye BILKUL SAME hai linear regression ke formula se:**
```
Linear Regression: y_hat = m×x + b
                         = w1×x1 + w2×x2 + ... + b

SAME CHEEZ!
```

**Ek single neuron abhi tak = Linear Regression!**
Koi difference nahi. Abhi tak.

---

### A6: Activation Function — Yahi Hai Asli Nayi Cheez!

**Problem:** z = 0.199 hai. But iska matlab kya hai?
- Ye negative bhi ho sakta hai (-5.2)
- Ye bahut bada bhi ho sakta hai (+1000)
- Ye probability nahi hai!

**Hume z ko ek "meaningful range" mein laana hai.**

Trading ke liye: Hume probability chahiye (0 to 1 ke beech).
"73% chance hai ki ye trade jeetega" — AISE chahiye!

**Sigmoid function yehi karta hai:**
```
output = sigmoid(z) = 1 / (1 + e^(-z))

Plug in z = 0.199:
  output = 1 / (1 + e^(-0.199))
         = 1 / (1 + e^(-0.199))

  e^(-0.199) = 0.8195  (calculator se ya yaad se)

  output = 1 / (1 + 0.8195)
         = 1 / 1.8195
         = 0.5496
```

**Output = 0.5496 = 54.96% win probability!**

> **YE TUJHE PATA HAI — tune sigmoid deep basics mein pada tha!**

**Poora picture ab tak:**
```
   x1 = 0.15 ──── ×0.5 ───→ 0.075 ─┐
                                      ├── ADD ──→ z = 0.199 ──→ sigmoid ──→ 0.5496
   x2 = 0.08 ──── ×0.3 ───→ 0.024 ─┤                          (activation)
                                      │
   bias = 0.1 ────────────────────→──┘

              WEIGHTED SUM                         SQUASH TO 0-1
```

**DONE! Ye hai EK NEURON ka pura kaam:**
1. Features aate hain (numbers)
2. Har feature ko weight se multiply karo
3. Sab jod do + bias
4. Sigmoid (ya koi aur activation) lagao
5. Output = probability!

---

### A7: Comparison — Logistic Regression IS a Single Neuron

```
╔═════════════════════════════╦═════════════════════════════════╗
║   LOGISTIC REGRESSION       ║   SINGLE NEURON                 ║
╠═════════════════════════════╬═════════════════════════════════╣
║ z = w1x1 + w2x2 + b        ║ z = w1x1 + w2x2 + b            ║
║ p = sigmoid(z)              ║ output = sigmoid(z)             ║
║ Loss = Log Loss             ║ Loss = Log Loss                 ║
║ dL/dw = (p-y)*x             ║ dL/dw = (output-y)*x           ║
║ w = w - lr * gradient       ║ w = w - lr * gradient           ║
╠═════════════════════════════╬═════════════════════════════════╣
║         100% SAME           ║         100% SAME               ║
╚═════════════════════════════╩═════════════════════════════════╝
```

**Ek single neuron = logistic regression. Literally. Character by character same.**

---

## PART A: QUIZ — Pehle ye answer kar (bina notes dekhe)

**Q1.** Agar x = [0.2, 0.5, 0.1] aur w = [1, -1, 2], b = 0.3, toh z = ?

**Q2.** z = 0 pe sigmoid ki value kya hoti hai?

**Q3.** Bias ka kya kaam hai — ek line mein?

**Q4.** Kya ek single neuron XOR gate solve kar sakta hai? (Haan/Nahi, reason batana)

> **Part B mein: Multiple neurons, layers, aur matrix notation — slowly.**
