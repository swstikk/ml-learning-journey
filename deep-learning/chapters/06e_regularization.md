# 1.7: Regularization in Neural Networks — Overfitting Rokna

> **NN bahut powerful hai — itna ki wo TRAINING DATA KA RATTA MAAR LETA HAI.**
> **Regularization = "ratta maarne se roko, SAMJHNA seekho!"**

---

## Problem: Overfitting Kya Hai NN Mein?

```
Training Accuracy: 99.5%    ← "Mujhe sab yaad hai!"
Test Accuracy:     62.0%    ← "Naya data? Pata nahi!"

Ye hai overfitting. Model ne training data MEMORIZE kar liya.
Generalize nahi kiya.
```

**NN mein overfitting kyun zyada hota hai?**
```
Linear Regression: ~10 parameters     → Hard to overfit
Random Forest:     ~thousands          → Some overfitting
Neural Network:    ~10,000 to MILLIONS → VERY easy to overfit!

Tera trading network (13→32→16→1): 993 parameters
But sirf 6,528 training samples!
993 parameters trying to memorize 6,528 patterns → OVERFIT RISK!
```

---

## Method 1: Dropout — "Random Neurons Bandh Karo"

### Concept:

```
Training ke dauran, har iteration mein randomly 20-50% neurons ko OFF kar do!

Normal forward pass:               With Dropout (p=0.5):
                                    
  h1 = 0.4  ──→                     h1 = 0.4  ──→  (ALIVE)
  h2 = 0.7  ──→   output            h2 = 0.0  ──X  (DROPPED! = 0)
  h3 = 0.2  ──→                     h3 = 0.2  ──→  (ALIVE)
  h4 = 0.9  ──→                     h4 = 0.0  ──X  (DROPPED! = 0)

Next iteration: DIFFERENT neurons drop!
  h1 = 0.0  ──X                     
  h2 = 0.7  ──→                     
  h3 = 0.0  ──X                     
  h4 = 0.9  ──→                     
```

### Kyun kaam karta hai?

```
Bina dropout:
  Network ek specific neuron pe DEPEND karta hai.
  "h2 = 0.7 hai toh WIN, nahi toh LOSS"
  → Agar test mein h2 thoda different ho → FAIL!

Dropout se:
  h2 kabhi hai kabhi nahi → network DUSRE neurons se bhi seekhta hai
  → Redundancy badhti hai → Robust model!
  
Analogy:
  Team mein 1 star player pe depend mat karo.
  Randomly star player ko bench karo training mein.
  Toh baaki players bhi improve honge!
  Match mein (test time) sab players khelenge → STRONGER team!
```

### Important: Test Time Pe Dropout OFF!

```
Training: Dropout ON  (randomly neurons off)
Testing:  Dropout OFF (sab neurons on)

But ek problem: Training mein 50% neurons the, test mein 100%.
Output values DOUBLE ho jayengi!

Fix: Test time pe outputs ko dropout rate se multiply karo:
  output_test = output × (1 - dropout_rate)

Ya (PyTorch automatic karta hai):
  Training mein alive neurons ke output ko 1/(1-p) se scale karo
  → Test time pe kuch change nahi karna padta!
```

### PyTorch mein:
```python
self.dropout = nn.Dropout(p=0.3)  # 30% neurons off har iteration
# Use between layers:
x = self.relu(self.fc1(x))
x = self.dropout(x)               # ← yahan lagao
x = self.fc2(x)
```

---

## Method 2: Early Stopping — "Jab Improve Hona Band Ho, Ruk Jao"

```
Epoch 1:   Train Loss = 0.65   Val Loss = 0.68
Epoch 10:  Train Loss = 0.42   Val Loss = 0.45
Epoch 50:  Train Loss = 0.15   Val Loss = 0.22   ← Val loss improving
Epoch 100: Train Loss = 0.05   Val Loss = 0.20   ← BEST val loss!
Epoch 150: Train Loss = 0.01   Val Loss = 0.25   ← Val loss INCREASING = OVERFIT!
Epoch 200: Train Loss = 0.003  Val Loss = 0.35   ← More overfit!

         Train Loss         Val Loss
  0.7 |*   *                *   *
  0.5 | *   *                *   *
  0.3 |   *   *                *   * ← diverge!
  0.1 |     * * * * *           *
  0.0 |           * * *       STOP HERE! (epoch ~100)
      └──────────────── epoch

EARLY STOPPING: Jab validation loss 10-20 epochs tak improve na ho → STOP!
Best model (epoch 100) save karo, baaki discard.
```

**PyTorch mein (manual):**
```python
best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(1000):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')  # save best
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

---

## Method 3: Batch Normalization — "Har Layer Ka Output Normalize Karo"

### Problem:

```
Layer 1 output: [0.001, 500, -3, 0.02]   ← WILDLY different scales!
Layer 2 ko ye input milta hai → training UNSTABLE!

Har layer ke output ki distribution badal jaati hai jab weights update hote hain.
Ye "Internal Covariate Shift" kehlata hai.
```

### Solution: BatchNorm

```
Har layer ke output ko normalize karo (mean=0, std=1):

  1. Batch ka mean nikalo:    mu = mean(outputs)
  2. Batch ka std nikalo:     sigma = std(outputs)
  3. Normalize karo:          x_norm = (x - mu) / sigma
  4. Scale+Shift (learnable): x_out = gamma * x_norm + beta
                              (gamma aur beta LEARN hote hain!)

Step 4 kyun? 
  Kyunki hamesha mean=0, std=1 rakhna ZARURI nahi.
  Network ko decide karne do ki optimal distribution kya ho.
  gamma aur beta ye flexibility dete hain.
```

### Benefits:
```
1. Training FASTER (6x speedup possible!)
2. Less sensitive to learning rate choice
3. Acts as mild regularization (batch statistics = noise)
4. Allows higher learning rates
```

### PyTorch mein:
```python
self.bn1 = nn.BatchNorm1d(32)   # 32 = hidden layer size
# Use BEFORE activation:
x = self.fc1(x)
x = self.bn1(x)      # ← normalize
x = self.relu(x)      # ← then activate
```

---

## Method 4: Weight Decay (L2 Regularization)

```
Tu ye Ridge Regression se jaanta hai!

Loss = Original Loss + lambda × sum(w²)

Weights ko CHHOTA rakhne ki penalty!
Bade weights = complex model = overfit
Chhote weights = simple model = generalize

PyTorch mein Adam ke saath built-in:
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
                                                          ^^^^^^^^^^^^^^^^
                                                          Ye hai L2 penalty!
```

---

## Kab Kya Use Karo — Practical Guide

```
╔══════════════════╦═══════════════════════╦═══════════════════════════╗
║ Method           ║ Kab Use Karo          ║ Typical Values            ║
╠══════════════════╬═══════════════════════╬═══════════════════════════╣
║ Dropout          ║ HAMESHA (default!)    ║ p=0.2 to 0.5             ║
║ Early Stopping   ║ HAMESHA               ║ patience=10-20 epochs    ║
║ BatchNorm        ║ Deep networks (3+)    ║ After linear, before act ║
║ Weight Decay     ║ When overfitting      ║ 0.0001 to 0.01           ║
║ Data Augmentation║ When less data         ║ (images/text transforms) ║
╚══════════════════╩═══════════════════════╩═══════════════════════════╝

MINIMUM: Dropout + Early Stopping HAMESHA lagao.
RECOMMENDED: Dropout + Early Stopping + BatchNorm.
```

---

## Trading Context

```
Tere 6,528 training samples + 993 parameters:
  → Dropout(0.3) ZARURI hai
  → Early Stopping ZARURI hai (patience=15)
  → BatchNorm optional but helps
  → Weight decay = 0.001 try karo

Network:
  Input(13) → Linear(32) → BatchNorm → ReLU → Dropout(0.3)
            → Linear(16) → BatchNorm → ReLU → Dropout(0.3)
            → Linear(1)  → Sigmoid
            
  Optimizer: Adam(lr=0.001, weight_decay=0.001)
  Early Stopping: patience=15
```

---

## Summary

```
OVERFITTING = memorize training, fail on test.

FIXES:
  Dropout:        Random neurons off (p=0.3)     → redundancy
  Early Stopping: Ruko jab val loss badhne lage   → right amount of training
  BatchNorm:      Normalize layer outputs          → stable, fast training
  Weight Decay:   Penalize large weights (L2)      → simpler model
  
  ALWAYS USE: Dropout + Early Stopping.
  
Phase 1 Theory: DONE! Ab PyTorch coding! 🔥
```
