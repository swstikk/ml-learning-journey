# PyTorch Step by Step — Bilkul Basic Se

> **Har cheez explain karenge. Koi line skip nahi.**
> **Pehle concept, phir code, phir hands-on.**

---

## PART 1: Epoch Kya Hai? (Pehle Ye Samjho)

### Yaad kar — Linear Regression mein kya kiya tha:

```
Tune haath se kiya tha:
  1. Data liya: 5 points (x, y)
  2. Line banai: y = mx + b
  3. Loss calculate kiya: SSE
  4. Gradient nikala: dL/dm, dL/db
  5. Weights update kiye: m = m - lr * gradient
  6. WAPAS step 3 pe gaya...
  7. ... ye loop 100 baar chala
```

**Ye loop = TRAINING LOOP.**
**1 baar pura dataset dekhna = 1 EPOCH.**

---

### Epoch Ka Matlab Number Se Samjho:

```
Tera data: 6,528 trades

EPOCH 1:
  Network ne 6,528 sare trades dekhe
  Sare trades pe loss calculate kiya
  Weights update kiye
  → EPOCH 1 COMPLETE

EPOCH 2:
  SAME 6,528 trades DOBARA dekhe (but weights update hue hain!)
  Loss ab thoda kam hoga (network kuch seekh gaya hai)
  Weights phir update kiye
  → EPOCH 2 COMPLETE

EPOCH 100:
  6,528 trades 100 baar dekhe ja chuke hain
  Network bahut kuch seekh gaya hai
  Loss bahut kam ho gayi hai
  → TRAINING COMPLETE!
```

> **SIMPLE: Epoch = Pura Dataset ek baar dikhana**
> 100 epochs = Dataset 100 baar dikhana = Network 100 baar seekhta hai

---

### Lekin Ek Saath Sab Data Nahi Dalte — Kyun?

```
Tera data: 6,528 trades
RAM mein sab ek saath? Fine.

Lekin soch agar 1 MILLION trades hote?
Ya 10 MILLION images hote (ImageNet)?

RAM = 16GB. 10M images = terabytes!
EK SAATH SAB NAHI AAL SAKTA!

SOLUTION: BATCH!
```

---

### Batch Kya Hai:

```
Pura data: 6,528 trades

Batch size = 64 (common choice)

Toh batches:
  Batch 1: trades 1-64
  Batch 2: trades 65-128
  Batch 3: trades 129-192
  ...
  Batch 102: trades 6,465-6,528   (6528 / 64 = 102 batches!)

EPOCH 1 = Batch 1 + Batch 2 + ... + Batch 102 = 6,528 trades dekhe!

Har BATCH pe:
  → Loss nikalo (sirf 64 trades pe)
  → Gradient nikalo
  → Weights update karo
  
1 EPOCH mein 102 updates hoti hain! (har batch pe ek update)
```

**Visualize:**
```
EPOCH 1:
  [Batch 1: 64 trades] → loss → gradient → UPDATE weights
  [Batch 2: 64 trades] → loss → gradient → UPDATE weights
  [Batch 3: 64 trades] → loss → gradient → UPDATE weights
  ...
  [Batch 102: 64 trades] → loss → gradient → UPDATE weights
  ↑ EPOCH 1 DONE (102 updates hue!)

EPOCH 2:
  [Batch 1: SAME 64 trades, but SHUFFLED] → loss → UPDATE
  ...
  ↑ EPOCH 2 DONE

... 100 epochs ...
```

> **KEY: Batch = Chhota chunk. Epoch = Sab batches ek baar.**
> **Adam updates har BATCH ke baad hota hai. 100 epochs mein 102×100 = 10,200 updates!**

---

### Adam aur Batches:

```
Tune pucha: "Adam mein kaise hota hai?"

SAME concept! Adam bhi gradient use karta hai.
Sirf update ka FORMULA alag hai (momentum + adaptive lr).

PROCESS:
  Batch 1 ka loss → Batch 1 ka gradient → ADAM update formula → weights update
  Batch 2 ka loss → Batch 2 ka gradient → ADAM update formula → weights update
  ...

Adam apne andar:
  m = "pichle gradients ka running average" (momentum)
  v = "pichle gradients² ka running average" (adaptive lr)
  
  Har batch ke baad ye dono update hote hain!
  Is wajah se Adam SMART hai — wo history yaad rakhta hai!
```

---

## PART 2: PyTorch Concepts — Ek Ek Karke

### 2.1 Tensor Kya Hai?

```python
import torch

# NumPy array:
import numpy as np
a = np.array([1.0, 2.0, 3.0])

# Pytorch Tensor (SAME cheez, different name):
t = torch.tensor([1.0, 2.0, 3.0])

# Dono mein operations same hain:
print(a * 2)     # [2. 4. 6.]
print(t * 2)     # tensor([2., 4., 6.])

# Tensor se NumPy:
a2 = t.numpy()   # [1. 2. 3.]

# NumPy se Tensor:
t2 = torch.from_numpy(a)
```

**Tensor = NumPy array. Bas different library ka.**
**Difference: Tensor GPU pe chal sakta hai aur gradients track karta hai.**

---

### 2.2 nn.Linear Kya Hai?

```
Tune manually kiya tha:
  z = w1*x1 + w2*x2 + b

nn.Linear YAHI karta hai! Bas automatic!

nn.Linear(in_features, out_features)
  in_features  = kitne inputs aate hain
  out_features = kitne outputs chahiye (= kitne neurons)

Example:
  nn.Linear(13, 32)
  → 13 inputs, 32 neurons
  → Automatically ek W matrix banata hai (32×13)
  → Automatically ek b vector banata hai (32,)
  → Forward: output = W @ input + b
```

```python
import torch.nn as nn

layer = nn.Linear(13, 32)

# Iske andar kya hai?
print(layer.weight.shape)   # torch.Size([32, 13]) ← W matrix!
print(layer.bias.shape)     # torch.Size([32]) ← b vector!

# Use karo:
x = torch.randn(13)         # 13 random inputs
z = layer(x)                # W @ x + b automatically!
print(z.shape)               # torch.Size([32]) ← 32 outputs!
```

> **nn.Linear = Ek Layer ka kaam. W matrix + b vector + multiply — sab automatic!**

---

### 2.3 Activation Functions in PyTorch:

```python
relu = nn.ReLU()
sigmoid = nn.Sigmoid()

x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])

print(relu(x))     # tensor([0.0, 0.0, 0.0, 0.5, 2.0])
print(sigmoid(x))  # tensor([0.12, 0.38, 0.50, 0.62, 0.88])
```

---

### 2.4 nn.Sequential — Layers Ko Chain Karo

```python
# Bina Sequential (verbose):
layer1 = nn.Linear(13, 32)
relu1  = nn.ReLU()
layer2 = nn.Linear(32, 16)
relu2  = nn.ReLU()
layer3 = nn.Linear(16, 1)
sig    = nn.Sigmoid()

# Phir forward mein:
def forward(x):
    x = layer1(x)
    x = relu1(x)
    x = layer2(x)
    x = relu2(x)
    x = layer3(x)
    x = sig(x)
    return x

# Ke saath Sequential (SAME cheez, clean):
model = nn.Sequential(
    nn.Linear(13, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

# Use:
x = torch.randn(13)
output = model(x)   # Data automatically sab layers se guzarta hai!
```

> **Sequential = Layers ki queue. Data andar jaata hai, ek ek layer se guzarta hai, output aata hai.**

---

### 2.5 Loss Function in PyTorch:

```python
criterion = nn.BCELoss()  # Binary Cross Entropy = Log Loss (tune padha tha!)

# Predicted probability:
pred = torch.tensor([0.53])   # model ka output
actual = torch.tensor([1.0])  # actual label (Win)

loss = criterion(pred, actual)
print(loss)   # tensor(0.6349)   ← Same formula! -log(0.53) = 0.63!
```

---

### 2.6 Optimizer in PyTorch:

```python
# Model ke sab weights aur biases track karta hai:
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Adam ke andar:
# model.parameters() = W1, b1, W2, b2, W3, b3 sab!
# lr = learning rate = 0.001 (default)
```

---

### 2.7 Training Loop — SIMPLE VERSION (First Samjho)

**Pehle sirf 1 sample ke liye:**

```python
# --- TERI MANUAL TRAINING LOOP (Matlab, concept) ---
#
# for 100 iterations:
#   1. input lo
#   2. output calculate karo (forward pass)
#   3. loss nikalo
#   4. gradient nikalo (dL/dw)
#   5. weights update karo (w = w - lr * grad)
#
# YAHI PyTorch mein:

for step in range(100):
    # Step 1+2: Forward pass
    output = model(x)
    
    # Step 3: Loss
    loss = criterion(output, y)
    
    # Step 4: Gradients CLEAR karo (pichla gradient hatao)
    optimizer.zero_grad()
    
    # Step 5: AUTOMATIC backprop (gradients calculate!)
    loss.backward()
    
    # Step 6: Weights update (Adam formula se)
    optimizer.step()
```

> **`optimizer.zero_grad()` kyun?**
> PyTorch gradients JODTA RAHA JATA HAI by default!
> Agar clear nahi kiya, toh:
>   Step 1 gradient: +0.04
>   Step 2 gradient: +0.04 + 0.03 = 0.07 (galat! sirf 0.03 chahiye tha!)
> Isliye har step se pehle ZERO karo!

---

## PART 3: Epoch + Batch Wala Loop — Ab Samajh Aayega

```python
# Pehle data ko batches mein todna:
from torch.utils.data import TensorDataset, DataLoader

# Maan lo:
X_train = torch.randn(6528, 13)  # 6528 trades, 13 features
y_train = torch.randint(0, 2, (6528, 1)).float()  # 6528 labels (0 ya 1)

# TensorDataset = X aur y ko saath bandho
dataset = TensorDataset(X_train, y_train)

# DataLoader = Data ko batches mein todte raho
loader = DataLoader(dataset, batch_size=64, shuffle=True)
#                            ^^^^^^^^^^^    ^^^^^^^^^^^^
#                            64 trades      Har epoch mein
#                            ek batch       order random karo!
```

**Now the full loop:**

```python
model = nn.Sequential(
    nn.Linear(13, 32), nn.ReLU(),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, 1),  nn.Sigmoid()
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(100):          # 100 baar pura dataset dekhna
    
    for batch_X, batch_y in loader:   # Ek ek batch uthao
        # batch_X = 64 trades (64, 13)
        # batch_y = 64 labels (64, 1)
        
        # SAME 4 steps as before, but ab 64 samples pe ek saath!
        output = model(batch_X)               # Forward (64 outputs)
        loss = criterion(output, batch_y)     # Loss (64 losses ka average)
        optimizer.zero_grad()                 # Clear old gradients
        loss.backward()                       # Backprop
        optimizer.step()                      # Update weights
    
    # Epoch khatam! Print karo progress:
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

**Kya ho raha hai step by step:**
```
EPOCH 1:
  Batch 1 (trades 0-63):
    → model(batch_X) = 64 predictions
    → loss = BCELoss average of 64
    → zero_grad → backward → step
    → weights thode update hue!
    
  Batch 2 (trades 64-127):
    → SAME, different trades
    → weights aur update!
    
  ... 102 batches ...
  
  EPOCH 1 DONE!
  Print: "Epoch 1: Loss = 0.6821"

EPOCH 2:
  (shuffle=True se) trades ki order random ho gayi!
  Batch 1 ab (trades 3421, 77, 6103, ....) ho sakta hai!
  
  ... 102 batches ...
  
  Print: "Epoch 2: Loss = 0.6544"  ← thoda kam!

EPOCH 100:
  Print: "Epoch 100: Loss = 0.2341"  ← bahut kam! Network seekh gaya!
```

---

## PART 4: model.train() aur model.eval() — Ye Kyun?

```python
model.train()   # Training mode
model.eval()    # Testing mode
```

**Dropout ka training vs testing farak yaad hai?**

```
Training mode (model.train()):
  - Dropout ON → Neurons randomly off
  - BatchNorm → Batch ka mean/std use karo

Testing mode (model.eval()):
  - Dropout OFF → Sab neurons on (full prediction!)
  - BatchNorm → Stored statistics use karo (batch nahi)

Agar ye switch nahi kiya:
  Test time pe bhi dropout chalega → galat random outputs → AUC bahut kam!
```

---

## PART 5: torch.no_grad() — Test Time Pe Gradients Mat Calculate Karo

```python
with torch.no_grad():
    predictions = model(X_test)
```

**Kyun?**
```
Training mein: Gradients chahiye (backprop ke liye)
Testing mein:  Gradients chahiye NAHI! (sirf prediction chahiye)

Bina torch.no_grad():
  PyTorch sab intermediate computations store karta hai (gradients ke liye)
  → DOUBLE MEMORY USE → Slow!

torch.no_grad() ke andar:
  PyTorch kuch store nahi karta
  → Fast! → Kam RAM!

RULE: Test/validation time pe HAMESHA torch.no_grad() use karo!
```

---

## PART 6: Ab Khud Likho — Guided Practice

**Task 1: Ek Simple Network Banao (ABHI KARO!)**

```python
import torch
import torch.nn as nn

# Step 1: Ek single sample banao
x = torch.tensor([0.15, 0.08])   # 2 features (SL dist, BB size)
y = torch.tensor([1.0])           # Actual: WIN

# Step 2: Network banao (2 inputs → 4 hidden → 1 output)
model = nn.Sequential(
    nn.Linear(2, 4),    # 2 inputs, 4 hidden neurons
    nn.ReLU(),
    nn.Linear(4, 1),    # 4 hidden, 1 output
    nn.Sigmoid()
)

# Step 3: Loss aur optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Step 4: Train karo 50 steps
for step in range(50):
    output = model(x)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 10 == 0:
        print(f"Step {step}: Prediction={output.item():.4f}, Loss={loss.item():.4f}")

# Expected output:
# Step 0:  Prediction=0.5234, Loss=0.6482  ← random start
# Step 10: Prediction=0.6123, Loss=0.4891  ← improving!
# Step 20: Prediction=0.7456, Loss=0.2934  ← better!
# Step 30: Prediction=0.8432, Loss=0.1710  ← much better!
# Step 40: Prediction=0.9012, Loss=0.1040  ← great!
# Step 49: Prediction=0.9345, Loss=0.0676  ← confident WIN prediction!
```

**Agar ye samajh aaya, toh sab samajh aaya!**

---

## PART 7: Concept Summary — Ek Table Mein

```
╔══════════════════════╦═══════════════════════════════════════════════╗
║ Cheez               ║ Matlab                                         ║
╠══════════════════════╬═══════════════════════════════════════════════╣
║ Epoch               ║ Pura dataset ek baar dekhna                   ║
║ Batch               ║ Dataset ka ek chhota chunk (e.g., 64 samples) ║
║ Tensor              ║ NumPy array (but for PyTorch)                  ║
║ nn.Linear(a, b)     ║ Layer: a inputs → b neurons (W @ x + b)       ║
║ nn.ReLU()           ║ ReLU activation (max(0, z))                    ║
║ nn.Sigmoid()        ║ Sigmoid activation (0-1 probability)           ║
║ nn.BCELoss()        ║ Log Loss (binary classification)               ║
║ optimizer.zero_grad ║ Pichle step ke gradients hatao                ║
║ loss.backward()     ║ AUTOMATIC backprop — sab gradients nikalo!    ║
║ optimizer.step()    ║ Gradients se weights update karo (Adam!)       ║
║ model.train()       ║ Dropout ON, BN train mode                     ║
║ model.eval()        ║ Dropout OFF, BN eval mode                     ║
║ torch.no_grad()     ║ Gradients mat nikalo (test time pe!)           ║
║ DataLoader          ║ Data ko batches mein automatically todte hai  ║
╚══════════════════════╩═══════════════════════════════════════════════╝
```

---

## NEXT STEP — Pehle Task 1 Run Karo

1. Ek `.py` file banao
2. Part 6 ka Task 1 code paste karo
3. Run karo
4. Output dekho — predictions 0.5 se 0.9+ tak jaati hain!
5. Bol mujhe kab run kiya — phir tera pura trading data pe karenge!
