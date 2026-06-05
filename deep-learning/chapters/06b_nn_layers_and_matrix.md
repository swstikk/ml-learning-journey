# Neural Networks — Part B: Multiple Neurons, Layers, Matrix

> **Part A mein ek neuron samjha. Ab BAHUT saare neurons.**
> **Yahan se "Neural Network" actually shuru hota hai.**

---

## PART B1: Kyun Ek Neuron Kaafi Nahi Hai?

### Ek Neuron Ki Limitation

Ek neuron ye karta hai:
```
z = w1×x1 + w2×x2 + b
output = sigmoid(z)
```

Ye equation ek **straight line** (ya plane) draw karta hai jo data ko do hisson mein baantna chahta hai.

```
  x2
   |      Loss Loss Loss
   |    Loss  Loss  Loss
   |  ──────────────────── ← Decision boundary (STRAIGHT LINE!)
   |    Win   Win   Win
   |  Win   Win   Win
   └──────────────────── x1
```

**Lekin agar data aisa ho:**
```
  x2
   |    Win   Loss  Win
   |  Loss   Win   Loss    ← Koi straight line se separate nahi hoga!
   |    Win   Loss  Win
   └──────────────────── x1
```

**Ek neuron = ek straight line = sirf linear patterns.**
Real trading KABHI linear nahi hota!

> **Solution: Bahut saare neurons lagao → complex curves ban sakti hain!**

---

## PART B2: Do Neurons — Kya Badalta Hai?

Ab soch: Ek ki jagah DO neurons rakh, dono SAME input le rahe hain.

```
                    Neuron 1
   x1 = 0.15 ─── w11=0.5 ──→ z1 = 0.5×0.15 + 0.3×0.08 + 0.1 = 0.199
             ╲               → h1 = sigmoid(0.199) = 0.5496
              ╲ w12=0.3
               ╲
                ╲
                 ╲
                  Neuron 2
   x2 = 0.08 ─── w21=-0.2 ─→ z2 = (-0.2)×0.15 + 0.8×0.08 + (-0.1) = -0.066
              ╱              → h2 = sigmoid(-0.066) = 0.4835
             ╱ w22=0.8
```

Wait — ye confusing lag raha hai. Ek ek karke samjho:

### Neuron 1 ka kaam:
```
Neuron 1 ke APNE weights hain: w11=0.5, w12=0.3, b1=0.1

z1 = w11 × x1 + w12 × x2 + b1
   = 0.5 × 0.15 + 0.3 × 0.08 + 0.1
   = 0.075 + 0.024 + 0.1
   = 0.199

h1 = sigmoid(0.199) = 0.5496
```

### Neuron 2 ka kaam (BILKUL INDEPENDENT, ALAG weights):
```
Neuron 2 ke APNE weights hain: w21=-0.2, w22=0.8, b2=-0.1

z2 = w21 × x1 + w22 × x2 + b2
   = (-0.2) × 0.15 + 0.8 × 0.08 + (-0.1)
   = -0.030 + 0.064 + (-0.1)
   = -0.066

h2 = sigmoid(-0.066) = 0.4835
```

### Kya observe kiya?

```
SAME input gaya (x1=0.15, x2=0.08) dono neurons mein.

Lekin ALAG weights hain:
  Neuron 1: w=[0.5, 0.3],   b=0.1   → output = 0.5496
  Neuron 2: w=[-0.2, 0.8],  b=-0.1  → output = 0.4835

ALAG answer aaya!
```

> **KEY INSIGHT:**
> Neuron 1 ne SL distance ko zyada importance di (w11=0.5).
> Neuron 2 ne BB size ko zyada importance di (w22=0.8).
> 
> Har neuron data ko ALAG angle se dekh raha hai!
> Neuron 1 = "SL specialist"
> Neuron 2 = "BB specialist"
> 
> Dono milke zyada information capture karte hain!

---

## PART B3: Drawing — 2 Neurons Saath Mein

```
                          ┌─────────────┐
   x1 = 0.15 ───w11=0.5──│  NEURON 1   │
              ╲           │  z1 = 0.199 │──→ h1 = 0.5496
   x2 = 0.08 ──w12=0.3───│  b1 = 0.1   │
                          └─────────────┘

                          ┌─────────────┐
   x1 = 0.15 ──w21=-0.2──│  NEURON 2   │
              ╲           │  z2 = -0.066│──→ h2 = 0.4835
   x2 = 0.08 ──w22=0.8───│  b2 = -0.1  │
                          └─────────────┘
```

**Har line ek weight hai. Har box ek neuron hai.**
**Har neuron independently apna kaam karta hai.**

---

## PART B4: Ye Do Neurons Milke = Ek "LAYER"

Jab bahut saare neurons SAME input lete hain aur PARALLEL mein kaam karte hain,
usko **ek layer** bolte hain.

```
           INPUT              HIDDEN LAYER (2 neurons)
           
   x1 ─────────────→ Neuron 1 → h1
         ╲       ╱
          ╲     ╱
           ╳  ╳      ← Sab inputs sab neurons se connected!
          ╱     ╲       ("Fully Connected" ya "Dense" layer)
         ╱       ╲
   x2 ─────────────→ Neuron 2 → h2
```

**"Fully Connected" matlab:**
- HAR input HAR neuron se connected hai
- x1 goes to BOTH Neuron 1 AND Neuron 2
- x2 goes to BOTH Neuron 1 AND Neuron 2
- 2 inputs × 2 neurons = 4 connections = 4 weights!

### Weight counting:

```
2 inputs, 2 neurons:
  Neuron 1: w11, w12, b1  → 3 parameters
  Neuron 2: w21, w22, b2  → 3 parameters
  Total: 6 parameters

General: n inputs, m neurons:
  Weights: n × m
  Biases:  m
  Total:   n×m + m = m×(n+1)
```

---

## PART B5: Ab MATRIX Notation — Kyun Chahiye?

### Bina Matrix:

```
Neuron 1: z1 = w11×x1 + w12×x2 + b1
Neuron 2: z2 = w21×x1 + w22×x2 + b2
```

2 neurons ke liye 2 lines likh di.
Agar 100 neurons hon? 100 lines likhega?
Agar 1000 neurons hon? 1000 lines?

**Matrix notation se ye SAB ek line mein ho jaata hai!**

### Matrix mein likhne ka tarika:

**Step 1: Sab weights ko ek table (matrix) mein daal do:**

```
W = [[w11, w12],      = [[0.5,  0.3],
     [w21, w22]]         [-0.2, 0.8]]

Row 1 = Neuron 1 ke weights
Row 2 = Neuron 2 ke weights
```

**Step 2: Inputs ko ek column mein daal do:**

```
x = [[x1],     = [[0.15],
     [x2]]        [0.08]]
```

**Step 3: Biases ko ek column mein daal do:**

```
b = [[b1],     = [[0.1],
     [b2]]        [-0.1]]
```

**Step 4: Matrix multiply karo!**

```
z = W × x + b

[[w11, w12],    [[x1],    [[b1],    [[w11×x1 + w12×x2 + b1],
 [w21, w22]]  ×  [x2]]  +  [b2]]  =  [w21×x1 + w22×x2 + b2]]

=  [[0.5×0.15 + 0.3×0.08 + 0.1],
    [(-0.2)×0.15 + 0.8×0.08 + (-0.1)]]

=  [[0.199],
    [-0.066]]
```

**SAME ANSWER as before! But ek line mein!**

```
z = W × x + b      ← Ye EK line = poora layer ka kaam!

Previously:
  z1 = w11×x1 + w12×x2 + b1    ← 2 lines chahiye thi
  z2 = w21×x1 + w22×x2 + b2

Now:
  z = W @ x + b                ← 1 line (@ = matrix multiply in Python)
```

> **Matrix notation = shortcut. Math SAME hai. Sirf likhne ka tarika efficient hai.**
> 
> 2 neurons ke liye farak nahi lagta.
> 100 neurons ke liye? 1 line vs 100 lines. Tab farak lagega!

### Matrix dimensions — ye yaad rakh:

```
W: (m × n)     m = neurons in this layer, n = inputs
x: (n × 1)     n = number of inputs
b: (m × 1)     m = neurons in this layer

z = W × x + b
(m×n) × (n×1) = (m×1)    + (m×1) = (m×1)

Hamara example:
W: (2 × 2)    2 neurons, 2 inputs
x: (2 × 1)    2 inputs
b: (2 × 1)    2 biases
z: (2 × 1)    2 outputs (one per neuron)
```

> **RULE:** Matrix multiply karne ke liye: pehle ka COLUMNS = doosre ki ROWS.
> (2×**2**) × (**2**×1) → 2 = 2 ✓ → result: (2×1)

---

## PART B6: Activation Har Neuron Pe Lagao

z nikala. Ab har z pe sigmoid lagao:

```
z = [[0.199],
     [-0.066]]

h = sigmoid(z) = [[sigmoid(0.199)],    = [[0.5496],
                   [sigmoid(-0.066)]]      [0.4835]]
```

**Activation "element-wise" lagti hai:**
Matlab har number pe INDEPENDENTLY lagti hai.

```
h = activation(z)    ← har z value pe separately activation function lagao
```

**Ab h = hidden layer ka output. Ye h aage next layer ka INPUT banega!**

---

## PART B7: Output Layer — Final Decision

Ab h1=0.5496 aur h2=0.4835 hamare paas hai.
In dono ko ek FINAL neuron mein bhejo jo WIN/LOSS decide kare.

```
Output neuron ke weights:
  v1 = 0.6    (h1 ka weight)
  v2 = -0.4   (h2 ka weight)
  b_out = 0.0

z_out = v1×h1 + v2×h2 + b_out
      = 0.6×0.5496 + (-0.4)×0.4835 + 0.0
      = 0.3298 + (-0.1934) + 0
      = 0.1364

output = sigmoid(z_out)
       = sigmoid(0.1364)
       = 1 / (1 + e^(-0.1364))
       = 1 / (1 + 0.8724)
       = 1 / 1.8724
       = 0.5340
```

**Final prediction: 53.4% win probability.**

---

## PART B8: Pura Network — Full Picture

```
┌──────────┐        ┌──────────────┐        ┌──────────────┐
│  INPUT   │        │ HIDDEN LAYER │        │ OUTPUT LAYER │
│          │        │              │        │              │
│ x1=0.15  │─0.5──→│ N1: z=0.199  │        │              │
│          │─0.3──→│ h1=0.5496    │──0.6──→│ z=0.1364     │
│          │       │              │        │ out=0.5340   │
│          │-0.2──→│ N2: z=-0.066 │        │              │
│ x2=0.08  │─0.8──→│ h2=0.4835   │─-0.4─→│              │
│          │        │              │        │              │
└──────────┘        └──────────────┘        └──────────────┘
  2 features          2 neurons               1 neuron
  (Layer 0)           (Layer 1)               (Layer 2)

  Weights: W1(2×2) + b1(2×1)      W2(1×2) + b2(1×1)
  Total parameters: 4+2 + 2+1 = 9
```

### Full Flow (data ka safar):

```
Step 1: x = [0.15, 0.08]                          ← Input features
Step 2: z1 = W1 @ x + b1 = [0.199, -0.066]        ← Weighted sum (Layer 1)
Step 3: h = sigmoid(z1) = [0.5496, 0.4835]         ← Activation (Layer 1)
Step 4: z2 = W2 @ h + b2 = [0.1364]                ← Weighted sum (Layer 2)
Step 5: output = sigmoid(z2) = [0.5340]             ← Activation (Layer 2)

DONE! Output = 0.5340 = "53.4% win probability"
```

**YE HAI FORWARD PASS. Data aage aage badhta gaya. Input se output tak.**

---

## PART B9: Layers Ki Chain — "Deep" Ka Matlab

```
1 Layer  (just sigmoid):    x → [Neuron] → output         = Logistic Regression
2 Layers (hidden+output):   x → [Layer1] → [Layer2] → out = "Shallow" NN
3 Layers:                   x → [L1] → [L2] → [L3] → out = "Deep" NN
5 Layers:                   x → [L1] → [L2] → [L3] → [L4] → [L5] → out

"DEEP Learning" = 3+ layers. That's it!
```

**Har layer ka output = next layer ka input.**

```
Layer 0 (Input):   x        (raw features)
Layer 1 (Hidden):  h1 = activation(W1 @ x + b1)
Layer 2 (Hidden):  h2 = activation(W2 @ h1 + b2)
Layer 3 (Output):  out = sigmoid(W3 @ h2 + b3)
```

> **Har layer data ko thoda TRANSFORM karti hai.**
> Layer 1: Raw features → "useful patterns"
> Layer 2: Patterns → "complex patterns"
> Layer 3: Complex patterns → "final decision"
> 
> Trading mein:
> Layer 1 seekhega: "SL chota + BB chota = tight conditions"
> Layer 2 seekhega: "Tight conditions + US session = high probability"
> Layer 3 decide karega: "WIN!"

---

## PART B10: ReLU — Sigmoid Ka Better Version for Hidden Layers

**Sigmoid ki problem hidden layers mein:**

```
Sigmoid derivative: max value = 0.25 (jab z=0)

3 hidden layers mein gradient multiply hota hai:
  0.25 × 0.25 × 0.25 = 0.0156

10 hidden layers mein:
  0.25^10 = 0.0000009536

Gradient basically ZERO ho gaya!
Weights update hi nahi honge!
MODEL KUCH SEEKHEGA HI NAHI!

Ye hai "Vanishing Gradient Problem"
```

**ReLU (Rectified Linear Unit) — simple fix:**

```
ReLU(z) = max(0, z)

If z > 0:  output = z,     derivative = 1     ← GRADIENT FULL PASS!
If z <= 0: output = 0,     derivative = 0     ← neuron "off"
```

**Visualize:**
```
  Sigmoid:                    ReLU:
  output                     output
  1 |     ──────────         │        ╱
    |    ╱                   │       ╱
    |   ╱                    │      ╱
  0.5──╱──                   │     ╱
    | ╱                      │    ╱
    |╱                       │   ╱
  0 ──────────── z           0──────────── z
     smooth S-curve             sharp angle at 0
```

**ReLU gradient:**
```
10 hidden layers, all positive z:
  1 × 1 × 1 × 1 × 1 × 1 × 1 × 1 × 1 × 1 = 1

  GRADIENT = 1! KUCH NAHI GHATA!
  Deep networks mein bhi weights seekhte hain!
```

### Kab Kya Use Karo:

```
Hidden layers:    ReLU     (gradient vanish nahi hota)
Output layer:     Sigmoid  (kyunki 0-1 probability chahiye)
Multi-class:      Softmax  (kyunki multiple class probabilities)

RULE: Hidden = ReLU. Output = Sigmoid/Softmax. Period.
```

### ReLU se forward pass recalculate karo:

```
Pehle sigmoid use kiya tha hidden layer mein:
  h1 = sigmoid(0.199) = 0.5496
  h2 = sigmoid(-0.066) = 0.4835

Ab ReLU use karo:
  h1 = ReLU(0.199)  = max(0, 0.199)  = 0.199    ← z positive tha → unchanged
  h2 = ReLU(-0.066) = max(0, -0.066) = 0.000     ← z negative tha → ZERO!

h2 = 0!  
Ye neuron DEAD hai is input ke liye!
Koi bhi information pass nahi kar raha!
Lekin agle input pe z2 positive ho sakta hai → tab alive hoga.
```

**ReLU wala forward pass:**
```
x = [0.15, 0.08]
z1 = W1 @ x + b1 = [0.199, -0.066]
h = ReLU(z1) = [0.199, 0.000]           ← ReLU used in hidden!
z2 = W2 @ h + b2 = 0.6×0.199 + (-0.4)×0.000 + 0 = 0.1194
output = sigmoid(z2) = 0.5298           ← Sigmoid still at output!
```

---

## PART B: FULL SUMMARY TABLE

```
╔════════════════╦══════════════════════════════════════════╗
║ Concept        ║ Explanation                              ║
╠════════════════╬══════════════════════════════════════════╣
║ Feature vector ║ x = [0.15, 0.08] — input numbers         ║
║ Weight         ║ "Kitna important hai" — per connection    ║
║ Bias           ║ "Default mood" — per neuron               ║
║ Weighted sum z ║ z = W @ x + b  (matrix multiply + bias)  ║
║ Activation     ║ ReLU(z) for hidden, sigmoid(z) for output║
║ Layer          ║ Group of neurons, same input, parallel    ║
║ Forward Pass   ║ Data ka safar: input → layers → output    ║
║ Matrix W       ║ Rows = neurons, Cols = inputs             ║
║ Deep           ║ 3+ layers = Deep Learning                 ║
║ Dead neuron    ║ ReLU output = 0, gradient = 0             ║
║ Vanishing grad ║ Sigmoid hidden → gradient → 0 → no learn ║
╚════════════════╩══════════════════════════════════════════╝
```

---

## PART B: QUIZ

**Q1.** 3 inputs, 5 hidden neurons. W1 ki shape kya hogi? z1 ki shape?

**Q2.** ReLU(-3.5) = ? ReLU(2.7) = ? ReLU(0) = ?

**Q3.** Hidden layer mein sigmoid kyun nahi use karte? 1 line mein.

**Q4.** Forward pass mein data kis direction mein jaata hai? (Aage/Peeche)

**Q5.** Agar hidden layer ka output [0.199, 0.000] hai aur output neuron ke weights [0.6, -0.4] hain aur bias=0 hai, toh z_out = ?

> **Ab tu ready hai Part C ke liye: Pura Forward Pass with numbers, phir Loss,**
> **phir 06_neural_networks_deep_math.md ke Step 4 (Backprop) pe seedha jaana.**
