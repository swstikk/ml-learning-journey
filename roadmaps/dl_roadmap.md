# Deep Learning — Complete Roadmap

> **Tera goal simple hai: NN se Transformer tak, jaldi aur solidly.**
> **Har cheez tere 4 goals se connected hai.**

---

## Tere 4 Goals — DL Kahan Lagega?

```
GOAL 1: ALGO TRADING (immediate)
  Needed: LSTM/GRU for price sequences, CNN for pattern recognition
  → Phase 3 mein milega

GOAL 2: ALPHAGENOOME-LIKE BIO AI (1-2 years)
  Needed: CNN (DNA motifs) + Transformer (long-range) + U-Net (multi-resolution)
  → Phase 2 + Phase 4 mein milega

GOAL 3: QUANTUM COMPUTING + ML (future)
  Needed: NN basics (variational circuits = parameterized NNs), autograd
  → Phase 1 mein milega

GOAL 4: LRM / LARGE REASONING MODELS (far future)
  Needed: Transformers + Attention + Scaling Laws + RL
  → Phase 4 + beyond
```

---

## Quick Answers to Your Questions

### DBSCAN vs K-Means?
DBSCAN is better at finding weird-shaped clusters and detecting outliers.
K-Means is simpler and faster.
**For trading:** K-Means is enough for market regime detection.
DBSCAN useful for anomaly detection (weird trades) — learn it in 20 min when you need it.
**Neither is blocking DL. Move forward.**

### Unsupervised Learning skip karna sahi hai?
**Mostly YES.** Here's the breakdown:
```
K-Means:     DONE (StatQuest dekh li) ← Enough
PCA/SVD:     DONE (math + StatQuest) ← Enough
DBSCAN:      SKIP for now (20 min job when needed)
Hierarchical: SKIP (rarely used in your domains)
t-SNE/UMAP:  SKIP for now (visualization tool, learn with DL projects)
Autoencoders: Will learn IN DL phase (Phase 2) ← natural place

Verdict: Unsupervised DONE ENOUGH. Start DL!
```

---

## Prerequisites Check — Kya Tu Ready Hai?

```
Linear Algebra basics        ✅ (Leonard lectures + Gilbert Strang SVD)
Gradient Descent             ✅ (Sigmoid lesson mein deeply kiya)
Chain Rule                   ✅ (Logistic regression backprop)
Loss Functions               ✅ (Log Loss, MSE — dono samjhe)
Sigmoid / Softmax            ✅ (Deep basics + softmax deep dive)
Classification Metrics       ✅ (AUC, F1, Confusion Matrix)
Python + NumPy + sklearn     ✅ (Trading project bana chuka)
PCA / SVD math               ✅ (StatQuest + Strang)

Missing (will learn IN the DL phases):
  PyTorch                    Phase 1 mein sikhega
  ReLU activation            Phase 1 mein (2 min concept)
  Adam optimizer             Phase 1 mein (just use it)
  Convolutions               Phase 2 mein
  Attention mechanism        Phase 4 mein

RESULT: 80% prerequisites DONE. Tu DL ke liye ready hai!
```

---

## PHASE 1: Neural Networks + Backprop + PyTorch (Week 1-2)

> **Ye sabse important phase hai. Iske bina kuch nahi hoga.**
> **Good news: Tu 60% already jaanta hai (sigmoid, gradient, chain rule)!**

### Theory Topics:

#### 1.1 Perceptron & Multi-Layer Network (Day 1)
```
Perceptron = Logistic Regression (tu jaanta hai!)
  Input → weights * inputs + bias → activation → output

Multi-Layer = Stack multiple perceptrons:
  Input Layer → Hidden Layer 1 → Hidden Layer 2 → Output
  
  Each "neuron" = sigmoid/ReLU(w*x + b)
  Layers stacked = "Deep" Learning!

Resources:
  StatQuest: "Neural Networks / Deep Learning" (19 min)
  3Blue1Brown: "But what is a Neural Network?" (19 min)
```

#### 1.2 Activation Functions (Day 1)
```
Sigmoid:  s(z) = 1/(1+e^-z)    ← Tu jaanta hai!
ReLU:     f(z) = max(0, z)      ← Naya! But simple.
Tanh:     f(z) = (e^z - e^-z)/(e^z + e^-z)
Softmax:  Multi-class output    ← Tu jaanta hai!

Why ReLU over Sigmoid in hidden layers?
  Sigmoid: Gradient vanishes (0.25 * 0.25 * 0.25... → near 0)
  ReLU: Gradient = 1 for positive values (no vanishing!)

Resources:
  StatQuest: "ReLU" (7 min)
```

#### 1.3 Forward Pass (Day 2)
```
Input → [Layer 1: W1*X + b1 → ReLU] → [Layer 2: W2*h1 + b2 → ReLU] → Output

Just matrix multiplications + activations chained together!
Tu matrix multiplication jaanta hai + sigmoid jaanta hai = forward pass jaanta hai!
```

#### 1.4 Loss Functions for NN (Day 2)
```
Binary Classification:  Log Loss (Cross-Entropy)  ← Tu jaanta hai!
Multi-class:           Categorical Cross-Entropy   ← Softmax + Log Loss
Regression:            MSE                          ← Tu jaanta hai!

Nothing new here for you!
```

#### 1.5 Backpropagation — THE KEY TOPIC (Day 3-4)
```
Ye logistic regression ke chain rule ka EXTENSION hai!

Logistic Regression (tu kiya tha):
  dL/dw = (p - y) * x     (ek layer ka gradient)

Neural Network:
  Same chain rule, but MULTIPLE LAYERS through:
  dL/dw3 → dL/dw2 → dL/dw1  (har layer ke weights update)

  "Error ko output se input tak propagate karo" = Backpropagation!

Resources:
  3Blue1Brown: "Backpropagation" (13 min) — MUST WATCH
  StatQuest: "Backpropagation" (10 min)
  3Blue1Brown: "Backpropagation calculus" (10 min) — math wala
```

#### 1.6 Optimizers (Day 4)
```
SGD:     Basic gradient descent (tu jaanta hai)
Adam:    SGD + momentum + adaptive learning rate
         → Just USE Adam. Don't derive. It works.

Resources:
  StatQuest: "Adam" (12 min)

Rule: Use Adam for everything. Default lr=0.001. Done.
```

#### 1.7 Regularization in NN (Day 5)
```
Dropout:       Randomly "turn off" 20% neurons during training
               → Forces network to not rely on any single neuron
BatchNorm:     Normalize layer outputs → faster, more stable training
Early Stopping: Stop when validation loss stops improving (tu jaanta hai!)

Resources:
  StatQuest: "Dropout" (7 min)
  StatQuest: "Batch Normalization" (15 min — optional, can learn later)
```

#### 1.8 PyTorch Introduction (Day 5-6)
```
PyTorch = "NumPy with GPU + automatic gradients"

Core concepts:
  torch.Tensor        → Like np.array, but on GPU
  autograd             → Automatic derivatives (no manual chain rule!)
  nn.Module            → Define your network
  DataLoader           → Feed data in batches
  loss.backward()      → Backprop in 1 line!
  optimizer.step()     → Update weights in 1 line!

Resources:
  PyTorch official: "60 Minute Blitz" tutorial
  https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
```

### Phase 1 Code Projects:

#### Project 1: NN From Scratch (NumPy, no frameworks)
```
- XOR problem solve karo (2 inputs → 1 output)
- Forward pass manually likh
- Backprop manually likh  
- See it LEARN — accuracy 50% → 100%
- THIS IS THE MOST IMPORTANT CODE YOU'LL EVER WRITE IN DL
```


#### Project 3: Bio-AI DNA Promoter NN
```
- Custom PyTorch Dataset on synthetic DNA sequences (60 bp)
- One-hot encode nucleotides (A, C, G, T)
- 3-layer MLP to classify Promoter vs Non-Promoter
- Handle mutation noise in regulatory motifs
```

### Phase 1 Milestones:
- [x] Can explain forward pass for 2-layer NN with ReLU
- [x] Backprop from scratch works (XOR)
- [x] MNIST 97%+ in PyTorch
- [x] Bio-AI DNA Promoter NN trained (99.8% Accuracy)

---

## PHASE 2: CNN — Convolutional Neural Networks (Week 3-4)

> **AlphaGenome ka foundation CNN hai. Trading mein candlestick pattern detection.**

### Theory Topics:

#### 2.1 Convolution Operation (Day 1)
```
Filter/Kernel slides across input → detects patterns!
  3x3 filter → edge detection
  5x5 filter → texture detection

1D Convolution → for sequences (DNA, price data!)
2D Convolution → for images

Resources:
  3Blue1Brown: "Convolutions in image processing" (21 min)
  StatQuest: "Convolutional Neural Networks" (22 min)
```

#### 2.2 Pooling, Stride, Padding (Day 2)
```
MaxPool:   Take maximum value in window → reduce size
Stride:    How much filter moves each step
Padding:   Add zeros around edges → keep same size

Architecture: Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Dense → Output
```

#### 2.3 Famous Architectures (Day 3 — overview only)
```
LeNet:     OG CNN (1998) — simple, good for learning
AlexNet:   ImageNet winner (2012) — deeper
VGG:       Very deep (16/19 layers) — simple blocks
ResNet:    Skip connections → 100+ layers without vanishing gradients
           IMPORTANT: This concept used EVERYWHERE (even Transformers!)

Resources:
  StatQuest: skip, just read a blog post comparing them
```

#### 2.4 1D CNN for Sequences (Day 4)
```
THIS IS KEY FOR BOTH YOUR GOALS:
  Trading: 1D CNN on price sequence → pattern detection
  Bio:     1D CNN on DNA sequence  → motif detection (AlphaGenome!)

Input: [p1, p2, p3, ..., p100]  (100 candle closes)
Conv1D filter slides across → detects patterns like "head and shoulders"
```

### Phase 2 Code Projects:

#### Project 1: Image Classifier
```
- CIFAR-10 dataset (10 classes: airplane, car, bird...)
- Simple CNN → 70%+ accuracy
- Add BatchNorm + Dropout → 80%+
```

#### Project 2: 1D CNN on Trading Data
```
- Sliding window: last 20 candles → predict next direction
- Input: (batch, 20, 5) → OHLCV for 20 candles
- Conv1D → Dense → Binary output
- Compare with your RF/XGBoost results!
```

### Phase 2 Milestones:
- [ ] CIFAR-10 > 75% accuracy
- [ ] Can explain convolution, pooling, stride
- [ ] 1D CNN on trading data trained
- [ ] Understand ResNet skip connections concept

---

## PHASE 3: RNN / LSTM — Sequence Models (Week 4-5)

> **TRADING SUPERPOWER! Price history = sequence = RNN/LSTM territory.**

### Theory Topics:

#### 3.1 RNN Basics (Day 1)
```
"Memory wala neural network"
  Normal NN: Input → Output (no memory of previous inputs)
  RNN: Input + Previous State → Output + New State (remembers!)

Trading: RNN remembers "pichle 50 candles ka pattern"

Problem: Vanilla RNN bhool jaata hai purani info (vanishing gradient)
Solution: LSTM!

Resources:
  StatQuest: "Recurrent Neural Networks (RNNs)" (14 min)
```

#### 3.2 LSTM — Long Short-Term Memory (Day 2-3)
```
3 Gates:
  Forget Gate: "Purani info bhoolun ya yaad rakhun?"
  Input Gate:  "Naya info kitna important hai?"
  Output Gate: "Kya output dun?"

Resources:
  StatQuest: "LSTM" (16 min) — MUST WATCH
  Colah's Blog: "Understanding LSTM Networks" — BEST explanation ever
  https://colah.github.io/posts/2015-08-Understanding-LSTMs/
```

#### 3.3 GRU — Simpler LSTM (Day 3)
```
GRU = LSTM lite (2 gates instead of 3)
Often same performance, faster training
Try both, pick what works better on your data
```

#### 3.4 Bidirectional & Stacked (Day 4)
```
Bidirectional: Read sequence both forward AND backward
Stacked: Multiple LSTM layers → deeper understanding

For DNA: Bidirectional LSTM = read both strands!
For trading: Usually unidirectional (can't see future)
```

### Phase 3 Code Projects:

#### Project 1: LSTM Price Predictor (THE BIG ONE!)
```
- YOUR real OHLCV data
- Input: 50 candles of features → predict next candle direction
- Features: Close, Volume, RSI, MACD (from your engine!)
- LSTM(hidden=64, layers=2) → Dense → Sigmoid → Buy/Sell
- THIS IS YOUR TRADING EDGE UPGRADE!
```

#### Project 2: Sequence Generation
```
- Character-level text generation
- Train on some text → generate new text
- Understand how sequence models "think"
```

### Phase 3 Milestones:
- [ ] Can explain LSTM gates (forget, input, output)
- [ ] LSTM trading predictor trained on your data
- [ ] Compare: RF vs XGBoost vs LSTM on trading data
- [ ] Understand bidirectional vs unidirectional choice

---

## PHASE 4: Transformers + Attention (Week 6-8)

> **THE KING. GPT, BERT, AlphaGenome, AlphaFold — SAB Transformer hai.**

### Theory Topics:

#### 4.1 Attention Mechanism (Day 1-2)
```
"Kaunse part pe zyada dhyan dena hai?"

Self-Attention:
  Query (Q): "Main kya dhundh raha hoon?"
  Key (K):   "Main kya offer kar raha hoon?"
  Value (V): "Meri actual information kya hai?"

  Attention(Q,K,V) = softmax(Q * K^T / sqrt(d_k)) * V

Resources:
  3Blue1Brown: "Attention in transformers, visually explained" — MUST
  StatQuest: "Transformer Neural Networks" (19 min)
  Jay Alammar: "The Illustrated Transformer" — BEST blog post EVER
  https://jalammar.github.io/illustrated-transformer/
```

#### 4.2 Multi-Head Attention (Day 3)
```
Instead of 1 attention: use 8 parallel attentions!
Each "head" learns different patterns:
  Head 1: Grammar patterns
  Head 2: Position patterns
  Head 3: Semantic patterns
  ...

For DNA: Different heads learn different motif interactions!
For trading: Different heads track different timeframe patterns!
```

#### 4.3 Positional Encoding (Day 3)
```
Transformers have no sense of order (unlike RNN)
Add position information: sin/cos functions at each position
```

#### 4.4 Full Transformer Architecture (Day 4-5)
```
Encoder: Input → Self-Attention → Feed-Forward → Output embedding
Decoder: Output → Masked Self-Attention → Cross-Attention → Feed-Forward

BERT = Encoder only (understanding)
GPT  = Decoder only (generation)
T5   = Full Encoder-Decoder (both)

AlphaGenome = CNN + Transformer Encoder + U-Net
```

#### 4.5 BERT & GPT Understanding (Day 6-7)
```
BERT: Predict masked words (fill-in-the-blank)
GPT:  Predict next word (autocomplete)

For Bio: DNA-BERT, Enformer use similar ideas on DNA "language"
For Trading: Time-series Transformers emerging!

Resources:
  Jay Alammar: "The Illustrated BERT"
  Jay Alammar: "The Illustrated GPT-2"
```

### Phase 4 Code Projects:

#### Project 1: Attention From Scratch
```
- Code Q, K, V attention in PyTorch (30 lines)
- Visualize attention weights
- See WHAT the model pays attention to
```

#### Project 2: Mini Transformer
```
- Build small transformer for text classification
- OR: Time-series transformer for trading
- This is the capstone!
```

#### Project 3 (Stretch): HuggingFace Pipeline
```
- Use pre-trained BERT for sentiment analysis
- Fine-tune on financial news → trading signal
- This connects ML + NLP + Trading!
```

### Phase 4 Milestones:
- [ ] Can explain Q, K, V attention from scratch
- [ ] Mini transformer coded and working
- [ ] Understand BERT vs GPT difference
- [ ] Can read AlphaGenome architecture and understand each block

---

## After DL — What Opens Up?

```
WITH DL COMPLETE, you can now learn:

For Trading:
  → Reinforcement Learning (RL agent that trades)
  → Time-Series Transformers (cutting edge)

For AlphaGenome/Bio:
  → Enformer (CNN + Transformer on DNA)
  → Protein structure (AlphaFold concepts)
  → U-Net for multi-resolution

For Quantum Computing:
  → Variational Quantum Circuits (= parameterized NNs on qubits)
  → Quantum ML (PennyLane library)

For LRM:
  → Scaling laws, RLHF, Chain-of-Thought
  → This needs Transformers + RL + massive compute understanding
```

---

## Resources Summary

```
VIDEO (Primary — watch in this order):
  3Blue1Brown: Neural Networks series (4 videos, ~1hr total) — INTUITION
  StatQuest: NN, ReLU, Backprop, LSTM, Transformer — CLEAR MATH
  Andrej Karpathy: "Neural Networks Zero to Hero" (YouTube) — CODING

BLOG (Read when stuck):
  Jay Alammar: Illustrated Transformer, BERT, GPT — BEST VISUALS
  Colah's Blog: LSTM Understanding — CLASSIC
  
CODE (Practice):
  PyTorch 60-Minute Blitz — OFFICIAL TUTORIAL
  fast.ai course — PRACTICAL (optional, if you want structured course)

BOOKS (Reference, don't read cover to cover):
  "Dive into Deep Learning" (d2l.ai) — FREE, interactive
```
