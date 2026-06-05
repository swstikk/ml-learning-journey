# 1.8: PyTorch — Coding Neural Networks (Hands On!)

> **Sab theory ho gaya. Ab CODE karna hai.**
> **PyTorch = NumPy + GPU + Automatic Backprop**

---

## PyTorch Core — 5 Cheezein Seekh, Sab Ho Jayega

### 1. Tensor = NumPy Array But Better

```python
import torch
import numpy as np

# NumPy:
a = np.array([1, 2, 3])

# PyTorch (exact same!):
t = torch.tensor([1, 2, 3])

# Convert:
t = torch.from_numpy(a)    # numpy → tensor
a = t.numpy()               # tensor → numpy

# Operations SAME hain:
t + 2           # [3, 4, 5]
t * 3           # [3, 6, 9]
t.mean()        # 2.0
t.shape         # torch.Size([3])

# Matrix:
W = torch.randn(2, 3)   # random (2x3) matrix
x = torch.randn(3, 1)   # random (3x1) vector
z = W @ x                # matrix multiply! Same as numpy!
```

**Difference from NumPy:** Tensor GPU pe chal sakta hai + gradients track karta hai!

### 2. Autograd = Automatic Backprop (MAGIC!)

```python
# Ye hai PyTorch ki sabse important feature!
# Tu manually backprop kiya tha. PyTorch KHUD karta hai!

x = torch.tensor(2.0, requires_grad=True)  # "Track gradients for this!"
y = x ** 2 + 3 * x + 1                      # y = x² + 3x + 1

y.backward()    # AUTOMATIC backprop! PyTorch ne sab derivatives nikal liye!

print(x.grad)   # dy/dx = 2x + 3 = 2(2) + 3 = 7.0
                 # CORRECT! Manually verify karo!
```

```python
# Neural network ke liye:
W = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
x = torch.randn(3)

z = W @ x + b
loss = z.sum()    # simplified loss

loss.backward()   # Sab gradients automatically!
print(W.grad)     # dL/dW — ye tu manually calculate karta tha!
print(b.grad)     # dL/db — ab ek line mein!
```

> **Tune 06_deep_math mein haath se dL/dW1 nikala tha. Chain rule, step by step.**
> **PyTorch mein: `loss.backward()` — DONE. 1 line. Same answer.**

### 3. nn.Module = Apna Network Define Karo

```python
import torch.nn as nn

class TradingNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Layers define karo:
        self.fc1 = nn.Linear(13, 32)    # 13 inputs → 32 hidden
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)    # 32 → 16 hidden
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 1)     # 16 → 1 output
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Forward pass define karo (data ka flow):
        x = self.fc1(x)        # Linear: W1 @ x + b1
        x = self.bn1(x)        # BatchNorm
        x = self.relu(x)       # ReLU activation
        x = self.dropout(x)    # Dropout (30%)
        
        x = self.fc2(x)        # Linear: W2 @ x + b2
        x = self.bn2(x)        # BatchNorm
        x = self.relu(x)       # ReLU
        x = self.dropout(x)    # Dropout
        
        x = self.fc3(x)        # Linear: W3 @ x + b3
        x = self.sigmoid(x)    # Sigmoid → probability!
        return x

# Use:
model = TradingNN()
x = torch.randn(32, 13)    # batch of 32 trades, 13 features each
output = model(x)           # forward pass!
print(output.shape)         # (32, 1) — 32 probabilities!
```

### 4. Training Loop = 6 Lines!

```python
# Setup:
model = TradingNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
criterion = nn.BCELoss()    # Binary Cross Entropy (Log Loss!)

# Training:
for epoch in range(100):
    model.train()                           # Dropout ON, BatchNorm train mode
    
    optimizer.zero_grad()                   # Step 0: Reset gradients
    predictions = model(X_train)            # Step 1: Forward pass
    loss = criterion(predictions, y_train)  # Step 2: Calculate loss
    loss.backward()                         # Step 3: Backprop (AUTOMATIC!)
    optimizer.step()                        # Step 4: Update weights
    
    # Validate:
    model.eval()                            # Dropout OFF, BatchNorm eval mode
    with torch.no_grad():                   # Don't track gradients
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train={loss:.4f} Val={val_loss:.4f}")
```

### 5. DataLoader = Data Feed Karo Batches Mein

```python
from torch.utils.data import TensorDataset, DataLoader

# Data prepare:
X_tensor = torch.FloatTensor(X_train_numpy)
y_tensor = torch.FloatTensor(y_train_numpy).unsqueeze(1)  # (N,) → (N,1)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Training with batches:
for epoch in range(100):
    for batch_X, batch_y in loader:   # 64 samples per batch
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
```

---

## Full Training Pipeline — Tera Trading Data

```python
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader

# ============ DATA ============
df = pd.read_csv('g:/plans/data/bpr_2203.csv')
FEATURES = [c for c in df.columns if c != 'is_win']

X = df[FEATURES].values.astype(np.float32)
y = df['is_win'].values.astype(np.float32)

# Time-aware split:
split = int(len(df) * 0.80)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# To tensors:
X_tr = torch.FloatTensor(X_train)
y_tr = torch.FloatTensor(y_train).unsqueeze(1)
X_te = torch.FloatTensor(X_test)
y_te = torch.FloatTensor(y_test).unsqueeze(1)

loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

# ============ MODEL ============
class TradingNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(16, 1),  nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = TradingNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
criterion = nn.BCELoss()

# ============ TRAIN ============
best_auc = 0
patience, counter = 15, 0

for epoch in range(200):
    model.train()
    for bx, by in loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
    
    # Validate:
    model.eval()
    with torch.no_grad():
        proba = model(X_te).numpy().flatten()
        auc = roc_auc_score(y_test, proba)
    
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), 'best_trading_nn.pt')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: AUC = {auc:.4f} (best={best_auc:.4f})")

print(f"\nBest AUC: {best_auc:.4f}")
print(f"Compare: RF was 0.8337, XGBoost was 0.8093")
```

---

## Phase 1: COMPLETE! Summary

```
06a: Single neuron (= logistic regression)
06b: Multiple neurons, layers, matrix notation
06c: Loss function, training loop
06_deep_math: Backpropagation (chain rule, all gradients)
06d: Optimizers (SGD → Adam)
06e: Regularization (Dropout, Early Stopping, BatchNorm)
06f: PyTorch (Tensors, autograd, nn.Module, training loop)

PHASE 1 = DONE!

YOU CAN NOW:
  ✅ Understand any NN architecture diagram
  ✅ Know what happens inside (forward + backward)
  ✅ Code and train NNs in PyTorch
  ✅ Apply to YOUR trading data
  
TOMORROW: CNN (Phase 2) → Trading pattern detection! 🔥
```
