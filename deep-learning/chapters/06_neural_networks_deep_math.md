# Neural Networks — Deep Math from Scratch
# (Same style as OLS derivation — pure logic, no magic)

> **Goal:** Tu khud ek neural network invent karega.
> Logistic regression jaanta hai? 90% kaam ho gaya.

---

## STEP 0: Ek Neuron ko Samjho (Warms Up!)

### Yaad kar: Logistic Regression

Tune logistic regression mein ye kiya tha:

```
z = w1*x1 + w2*x2 + w3*x3 + b        (linear combination)
p = sigmoid(z) = 1 / (1 + e^(-z))     (probability)
Loss = -[y*log(p) + (1-y)*log(1-p)]   (log loss)
```

**Ek neuron = bilkul yehi hai!**

```
x1 ──w1──┐
x2 ──w2──┤ → z = w1x1 + w2x2 + w3x3 + b → sigmoid(z) → output
x3 ──w3──┘
          b (bias)
```

Ye EK neuron hai. Simple.

**Problem:** Ek neuron sirf LINEAR patterns seekh sakta hai.
Agar data non-linear hai (aur real world HAMESHA non-linear hai)? Fail!

**Solution:** Bahut saare neurons, bahut saari layers!

---

## STEP 1: Network Architecture — Layers ka Logic

### Ek Simple 2-Layer Network (Trade Example):

**Goal:** Predict karo ki trade win hoga ya nahi.

```
INPUT LAYER          HIDDEN LAYER          OUTPUT LAYER
  (features)           (2 neurons)           (1 neuron)

  x1 (sl_dist)  ──── w11 ────┐
                              ├── h1 ──── v1 ──┐
  x2 (bb_size)  ──── w21 ────┘                  ├── output (win probability)
                ──── w12 ────┐                  │
                              ├── h2 ──── v2 ──┘
                ──── w22 ────┘

x1 = SL distance (e.g., 0.15)
x2 = BB size     (e.g., 0.08)
h1, h2 = hidden neurons (middle layer)
output = final prediction
```

### Notation Setup (ek baar set karo, phir yaad rakho):

```
n_0 = 2   (input features: x1, x2)
n_1 = 2   (hidden layer neurons: h1, h2)
n_2 = 1   (output neuron)

Weights:
  W1 = matrix of shape (n_1 x n_0) = (2 x 2)
       W1[j][i] = weight from input i to hidden neuron j

  W2 = matrix of shape (n_2 x n_1) = (1 x 2)
       W2[k][j] = weight from hidden j to output k

Biases:
  b1 = vector of shape (n_1,) = (2,)   → one per hidden neuron
  b2 = vector of shape (n_2,) = (1,)   → one per output neuron
```

---

## STEP 2: Forward Pass — Information Aage Badhna

Ye "prediction karna" ka process hai.

### Manually ek example karo:

**Input:** Trade ke features
```
x1 = 0.15  (SL distance)
x2 = 0.08  (BB size)
```

**Weights (randomly initialized):**
```
W1 = [[0.5,  0.3],    → w11=0.5, w12=0.3 (hidden neuron 1 ke weights)
      [-0.2, 0.8]]    → w21=-0.2, w22=0.8 (hidden neuron 2 ke weights)

b1 = [0.1, -0.1]      → bias for h1 and h2

W2 = [[0.6, -0.4]]    → output neuron ke weights
b2 = [0.0]            → output bias
```

### Layer 1: Input → Hidden

**Neuron h1 ka pre-activation (z):**
```
z1_1 = w11*x1 + w12*x2 + b1_1
     = 0.5*0.15 + 0.3*0.08 + 0.1
     = 0.075 + 0.024 + 0.1
     = 0.199
```

**Neuron h2 ka pre-activation (z):**
```
z1_2 = w21*x1 + w22*x2 + b1_2
     = (-0.2)*0.15 + 0.8*0.08 + (-0.1)
     = -0.03 + 0.064 - 0.1
     = -0.066
```

**Activation function apply karo — ReLU:**
```
ReLU(z) = max(0, z)

h1 = ReLU(z1_1) = ReLU(0.199) = 0.199   (positive → unchanged)
h2 = ReLU(z1_2) = ReLU(-0.066) = 0.000  (negative → becomes 0!)
```

> **Why ReLU?**
> Sigmoid ki problem: gradient bahut chota ho jaata hai (0 to 0.25 max).
> Multiple layers → gradient aur chota → basically 0 → kuch seekh nahi!
> ReLU: gradient = 1 for positive → no shrinking!

### Layer 2: Hidden → Output

```
z2_1 = v1*h1 + v2*h2 + b2
     = 0.6*0.199 + (-0.4)*0.000 + 0.0
     = 0.1194 + 0 + 0
     = 0.1194
```

**Output activation — Sigmoid (kyunki binary classification):**
```
output = sigmoid(z2_1) = 1 / (1 + e^(-0.1194))
       = 1 / (1 + e^(-0.1194))
       = 1 / (1 + 0.887)
       = 1 / 1.887
       = 0.530
```

**Prediction: 53% win probability.**

Actual: Let's say y = 1 (this trade WAS a win).

---

## STEP 3: Loss Function — Kitna Galat Hai?

Binary Cross-Entropy (Log Loss) — tu jaanta hai:

```
L = -[y * log(p) + (1-y) * log(1-p)]

y = 1 (actual win)
p = 0.530 (predicted)

L = -[1 * log(0.530) + (1-1) * log(1-0.530)]
  = -[1 * log(0.530) + 0]
  = -log(0.530)
  = -(-0.635)
  = 0.635
```

**Loss = 0.635** — kaafi high hai. Model ne 53% diya tha, but ye clearly win tha.
Weights update karne ki zaroorat hai!

---

## STEP 4: Backpropagation — THE KEY DERIVATION

### Concept: Error ko peeche ki taraf propagate karo

**Goal:** dL/dW1, dL/db1, dL/dW2, dL/db2 nikalna
(Loss kitna change hoga agar hum har weight ko thoda change karein?)

**Tool:** Chain Rule

```
Chain Rule reminder:
  If y = f(g(x))
  then dy/dx = (dy/dg) * (dg/dx)   ← multiply derivatives

In network:
  L depends on output
  output depends on z2
  z2 depends on W2, h
  h depends on z1
  z1 depends on W1, x

So: dL/dW1 = (dL/doutput) * (doutput/dz2) * (dz2/dh) * (dh/dz1) * (dz1/dW1)

Yahi hai backpropagation!
```

### OUTPUT LAYER se shuru karo (reverse direction):

#### Step 4A: dL/d(output) — Loss ka output ke saath derivative

```
L = -[y*log(p) + (1-y)*log(1-p)]     where p = output

dL/dp = -[y/p - (1-y)/(1-p)]
      = -[y*(1-p) - (1-y)*p] / [p*(1-p)]
      = -[y - yp - p + yp] / [p*(1-p)]
      = -[y - p] / [p*(1-p)]
      = (p - y) / [p*(1-p)]

Hamare values mein:
dL/dp = (0.530 - 1) / [0.530 * (1 - 0.530)]
      = -0.470 / [0.530 * 0.470]
      = -0.470 / 0.249
      = -1.887
```

#### Step 4B: d(output)/dz2 — Sigmoid ka derivative

```
output = sigmoid(z2) = 1/(1 + e^(-z2))

Sigmoid ka derivative (tune logistic regression mein kiya tha!):
  d(sigmoid)/dz = sigmoid(z) * (1 - sigmoid(z))
                = output * (1 - output)

d(output)/dz2 = 0.530 * (1 - 0.530)
              = 0.530 * 0.470
              = 0.249
```

#### Step 4C: dL/dz2 — Chain Rule!

```
dL/dz2 = (dL/d(output)) * (d(output)/dz2)
        = (-1.887) * (0.249)
        = -0.470

SHORTCUT (important!):
  dL/dz2 = output - y
          = 0.530 - 1
          = -0.470

This ALWAYS simplifies to (output - y) for sigmoid + log loss!
SAME as logistic regression! Just the last layer.
```

#### Step 4D: dL/dW2 — Output layer weights ka gradient

```
z2 = v1*h1 + v2*h2 + b2

dz2/dv1 = h1 = 0.199
dz2/dv2 = h2 = 0.000

dL/dv1 = (dL/dz2) * (dz2/dv1) = (-0.470) * 0.199 = -0.0935
dL/dv2 = (dL/dz2) * (dz2/dv2) = (-0.470) * 0.000 =  0.0000
dL/db2 = (dL/dz2) * 1          = (-0.470) * 1     = -0.470

So: W2 gradient = [-0.0935, 0.0000]
    b2 gradient = [-0.470]

Intuition:
  v1 ka gradient negative → v1 badhana chahiye (taaki loss kam ho)
  h2 = 0 tha → v2 se koi contribution nahi → gradient 0
```

#### Step 4E: dL/dh — Hidden layer outputs ka gradient

```
z2 = v1*h1 + v2*h2 + b2

dz2/dh1 = v1 = 0.6
dz2/dh2 = v2 = -0.4

dL/dh1 = (dL/dz2) * (dz2/dh1) = (-0.470) * 0.6  = -0.282
dL/dh2 = (dL/dz2) * (dz2/dh2) = (-0.470) * (-0.4) = 0.188
```

#### Step 4F: dL/dz1 — ReLU ka derivative (IMPORTANT!)

```
h1 = ReLU(z1_1)

ReLU derivative:
  d(ReLU)/dz = 1   if z > 0
               0   if z <= 0

Hamare values:
  z1_1 = 0.199 > 0  → dh1/dz1_1 = 1
  z1_2 = -0.066 < 0 → dh2/dz1_2 = 0  ← DEAD NEURON! h2=0 tha → gradient bhi 0!

dL/dz1_1 = (dL/dh1) * (dh1/dz1_1) = (-0.282) * 1 = -0.282
dL/dz1_2 = (dL/dh2) * (dh2/dz1_2) = (0.188) * 0  =  0.000
```

> **"Dead Neuron" concept:**
> h2 ka z = -0.066 tha → ReLU ne 0 kar diya → gradient 0 → weights update nahi hogi!
> h2 literally "soya hua" hai is example mein.

#### Step 4G: dL/dW1 — Input layer weights ka gradient

```
z1_1 = w11*x1 + w12*x2 + b1_1

dz1_1/dw11 = x1 = 0.15
dz1_1/dw12 = x2 = 0.08

dL/dw11 = (dL/dz1_1) * x1 = (-0.282) * 0.15 = -0.0423
dL/dw12 = (dL/dz1_1) * x2 = (-0.282) * 0.08 = -0.0226
dL/db1_1 = (dL/dz1_1) * 1 = -0.282

z1_2 = w21*x1 + w22*x2 + b1_2
dL/dw21 = (dL/dz1_2) * x1 = 0 * 0.15 = 0
dL/dw22 = (dL/dz1_2) * x2 = 0 * 0.08 = 0
dL/db1_2 = 0
```

---

## STEP 5: Weight Update — Learning!

```
Gradient Descent formula:
  new_weight = old_weight - learning_rate * gradient

Learning rate (eta) = 0.1  (typically 0.001-0.1)

W1 updates:
  w11: 0.5   - 0.1*(-0.0423) = 0.5   + 0.00423 = 0.50423
  w12: 0.3   - 0.1*(-0.0226) = 0.3   + 0.00226 = 0.30226
  w21: -0.2  - 0.1*(0)       = -0.2             = -0.2
  w22: 0.8   - 0.1*(0)       = 0.8              = 0.8
  b1_1: 0.1  - 0.1*(-0.282)  = 0.1   + 0.0282  = 0.1282
  b1_2: -0.1 - 0.1*(0)       = -0.1             = -0.1

W2 updates:
  v1:  0.6   - 0.1*(-0.0935) = 0.6   + 0.00935 = 0.60935
  v2:  -0.4  - 0.1*(0)       = -0.4             = -0.4
  b2:  0.0   - 0.1*(-0.470)  = 0.0   + 0.047   = 0.047
```

**Now do forward pass again with new weights → lower loss!**
Repeat 1000s of times = training!

---

## STEP 6: Ek Aur Example — 3-Layer Network

**Problem:** XOR gate banana (classic problem jo 1 neuron solve nahi kar sakta)

```
XOR Truth Table:
  x1=0, x2=0 → 0
  x1=0, x2=1 → 1
  x1=1, x2=0 → 1
  x1=1, x2=1 → 0

Notice: NOT linearly separable!
  (Cannot draw one straight line to separate 0s and 1s)
  Linear model (logistic regression) = FAIL
  Neural Network = WORKS!
```

**Architecture:**
```
n_0 = 2 inputs
n_1 = 2 hidden neurons (with ReLU)
n_2 = 1 output (with Sigmoid)
```

**Manually solve ONE forward pass:**

Input: x1=1, x2=1 → Expected output: 0

```
# Trained weights (after convergence):
W1 = [[1,  1],     # h1 fires when EITHER input is 1
      [1,  1]]     # h2 also responds to both
b1 = [0, -1]       # h2 has negative bias!

W2 = [[1, -2]]     # Output adds h1, subtracts 2*h2
b2 = [0]

# Forward pass:
z1_1 = 1*1 + 1*1 + 0 = 2    → h1 = ReLU(2) = 2
z1_2 = 1*1 + 1*1 + (-1) = 1 → h2 = ReLU(1) = 1

z2 = 1*2 + (-2)*1 + 0 = 2 - 2 = 0
output = sigmoid(0) = 0.5

Hmm... not perfect. But with more training iterations, network converges!
```

> **Why this works:**
> h1 = "at least one input is 1" (OR gate behavior)
> h2 = "both inputs are 1" (AND gate behavior)
> Output = h1 - 2*h2 = OR - 2*AND = XOR! Beautiful!

---

## STEP 7: General Formulas — Pattern Samjho

### Any network ke liye:

```
FORWARD PASS (Layer l):
  z[l] = W[l] @ a[l-1] + b[l]        (@ = matrix multiply)
  a[l] = activation(z[l])             (ReLU for hidden, sigmoid for output)

BACKWARD PASS:
  delta[L] = a[L] - y                 (output layer error, sigmoid+log loss shortcut)
  
  delta[l] = (W[l+1].T @ delta[l+1]) * ReLU_derivative(z[l])  (hidden layers)
  
  dW[l] = delta[l] @ a[l-1].T        (weight gradient)
  db[l] = delta[l]                    (bias gradient)

UPDATE:
  W[l] = W[l] - lr * dW[l]
  b[l] = b[l] - lr * db[l]
```

### Matrix dimensions check karna seekh (critical!):

```
Example: 3-layer network
  n_0 = 2, n_1 = 4, n_2 = 3, n_3 = 1

  W1: (4 x 2)     z1 = W1 @ x → (4,)
  W2: (3 x 4)     z2 = W2 @ a1 → (3,)
  W3: (1 x 3)     z3 = W3 @ a2 → (1,)

RULE: W[l] has shape (n[l] x n[l-1])
  Rows = neurons in current layer
  Cols = neurons in previous layer
```

---

## STEP 8: Teen Networks Manually Banana — Teri Practice

### Network 1: AND Gate (Easiest)

```
Truth Table:
  0, 0 → 0
  0, 1 → 0
  1, 0 → 0
  1, 1 → 1

TRY KARO:
  Architecture: 2 inputs, 1 output, NO hidden layer (logistic regression!)
  
  Start with W = [0.5, 0.5], b = -0.8
  
  Forward pass all 4 inputs.
  Calculate loss for each.
  Calculate gradient.
  Update weights.
  Repeat 10 times manually.
  
  See if output for (1,1) > 0.5 and rest < 0.5
```

### Network 2: 2-Layer — Your Trading Mini-Example

```
Use the exact example from STEP 2 above.

TASK:
  1. Complete the full forward pass (you have it above)
  2. Calculate loss (done above: 0.635)
  3. Calculate ALL gradients (done above)
  4. Update ALL weights (done above)
  5. Do a SECOND forward pass with new weights
  6. Is loss less than 0.635? (It should be!)
```

### Network 3: 3-Layer — XOR Challenge

```
Architecture:
  n_0 = 2, n_1 = 3, n_2 = 1
  
  Random initialization:
  W1 = [[0.1, -0.2], [0.4, 0.3], [-0.1, 0.5]]
  b1 = [0, 0, 0]
  W2 = [[0.3, -0.1, 0.2]]
  b2 = [0]
  
  Input: x1=0, x2=1 → Expected: 1
  
  TASK: Full forward pass → loss → all gradients → update
  This will take ~30 min manually. Do it!
  
  After doing it: code it in NumPy to verify!
```

---

## STEP 9: Dimensions aur Batch Training

Real training mein ek sample nahi, BAHUT SAARE samples ek saath!

```
Batch size B = 32 (32 trades ek saath process)

X: (B x n_0) = (32 x 2)    → 32 trades, 2 features each

Layer 1:
  z1 = X @ W1.T + b1        → (32 x n_1) = (32 x 4)
  a1 = ReLU(z1)             → (32 x 4)

Layer 2:
  z2 = a1 @ W2.T + b2       → (32 x n_2) = (32 x 1)
  output = sigmoid(z2)       → (32 x 1)

Loss:
  L = mean(-[y*log(p) + (1-y)*log(1-p)])   ← mean over batch!
```

> **Why batches?**
> Full dataset ek saath = accurate gradient but SLOW (RAM bhi bharega)
> 1 sample = noisy gradient but fast
> Batch of 32-256 = balance! This is "Mini-batch Gradient Descent"

---

## STEP 10: Weight Initialization — Kyun Random?

```
Problem 1: Agar sab weights = 0?
  z1 = W1 @ x + b1 = 0 @ x + 0 = [0, 0, 0, 0]
  All hidden neurons compute SAME thing!
  Gradients SAME!
  Weights update SAME way!
  → All neurons stay identical FOREVER → Useless!
  This is called "Symmetry Problem"

Problem 2: Agar weights bahut bade?
  z values = large
  Sigmoid: sigma(100) ≈ 1, sigma(-100) ≈ 0
  Gradient ≈ 0 → Nothing learns!

Solution: Small random initialization!

  Xavier (for Sigmoid/Tanh):
    W ~ Uniform(-1/sqrt(n_in), 1/sqrt(n_in))
  
  He (for ReLU):
    W ~ Normal(0, sqrt(2/n_in))
    (Slightly larger because ReLU kills half the neurons)
```

---

## SUMMARY — Pure Picture

```
FORWARD PASS (Prediction):
  x → [W1,b1 → ReLU] → [W2,b2 → ReLU] → [WL,bL → Sigmoid] → output

LOSS:
  L = -[y*log(output) + (1-y)*log(1-output)]

BACKWARD PASS (Learning):
  delta_L = output - y                              (output layer)
  dWL = delta_L @ a_{L-1}.T                        (output weights)
  
  delta_{l} = (W_{l+1}.T @ delta_{l+1}) * ReLU'   (hidden layers)
  dW_l = delta_l @ a_{l-1}.T                       (hidden weights)

UPDATE:
  W = W - lr * dW
  b = b - lr * db

REPEAT 1000s of times → MODEL LEARNS!
```

---

## Ek Baar Aur Sab Ek Jagah — Tera Trading Network

```
Tere bpr_2203.csv ka setup:

Input Layer:    13 neurons (13 features: sl_dist, bb_size, tp_used, etc.)
Hidden Layer 1: 32 neurons (ReLU)
Hidden Layer 2: 16 neurons (ReLU)
Output Layer:   1 neuron  (Sigmoid → win probability)

Parameters count:
  W1: 32 * 13 = 416
  b1: 32
  W2: 16 * 32 = 512
  b2: 16
  W3: 1 * 16 = 16
  b3: 1
  TOTAL: 993 parameters!

RF mein: ~300 trees, each with splits = thousands of parameters
Network mein: 993 parameters, but can capture non-linear relationships!

Train this on your 6,528 trades,
Test on last 1,633 trades.
Compare with RF AUC = 0.83!
```

> **Tu ab SAMJH sakta hai:**
> - Kyun network "learns" (gradient descent on loss)
> - Kyun layers zaroori hain (non-linearity)
> - Kyun ReLU better hai (gradient vanishing)
> - Kyun initialization important hai (symmetry breaking)
> - Backprop ka har step step kya ho raha hai
>
> **Teri understanding: Researcher level!**
