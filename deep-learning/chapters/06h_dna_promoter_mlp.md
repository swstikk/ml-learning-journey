# Step-by-Step Bio-AI Project: DNA Promoter Classifier (PyTorch MLP)
===================================================================

> **Target:** DNA sequence ko dekh kar predict karna ki kya wo ek **Promoter** (gene transcription starts here) hai ya nahi.
> **Style:** Hinglish, super deep math + code explanations, step-by-step logic, no skipping.

Is project mein hum ek biological problem solve karenge jo **AlphaGenome** aur bioinformatics ka absolute foundation hai! Aur iske sath hum PyTorch ke core features jaise **Custom Datasets**, **One-Hot Encoding** aur **MLPs** ko code level pe seekhenge.

---

## 1. Biology Primer: DNA Promoter Kya Hai?

Humari body ka genetic code DNA mein written hota hai. DNA chaar bases (nucleotides) se bana hota hai:
* **A** (Adenine)
* **C** (Cytosine)
* **G** (Guanine)
* **T** (Thymine)

Maan lo DNA ek lambi string hai: `ATGCTAGCTAGCT...`

Pure DNA mein har jagah genes (proteins banane ki instructions) nahi hote. Gene start hone se theek pehle ek control region hota hai jise **Promoter** kehte hain. 
Promoter cell ko batata hai: *"Arey! Yahan se transcription shuru karo, yahan gene hai!"*

### Motifs (DNA ke patterns)
Promoters mein specific patterns hote hain jinhe biological language mein **Motifs** kehte hain. Do mashhoor motifs hain:
1. **Pribnow Box (TATA Box):** Iska sequence standard `TATAAT` hota hai, aur ye gene ke start site se theek pehle aata hai.
2. **-35 Region:** Iska sequence generally `TTGACA` hota hai.

Humara model DNA sequences ko dekhkar seekhega ki kis sequence mein ye features/motifs hain aur use **Promoter (Class 1)** classify karega, aur jisme nahi hai use **Non-Promoter (Class 0)** classify karega.

---

## 2. Representation: DNA ko Computer Kaise Samjhe?

Neural network strings (`"A", "C", "G", "T"`) ko directly process nahi kar sakta. Hamein characters ko numbers mein badalna padega.

### Q: "Hum A=1, C=2, G=3, T=4 kyun nahi likh sakte?"
**A:** Agar hum A=1 aur G=3 rakhenge, to network sochega ki $G$ is 3 times bigger/more important than $A$, ya fir $C (2)$ aur $T (4)$ ka average $G (3)$ hai. Lekin biology mein A, C, G, T bas alag categories hain, unme koi numerical order ya magnitude nahi hota!

Isliye hum use karte hain **One-Hot Encoding**:

$$\text{A} \rightarrow [1, 0, 0, 0]$$
$$\text{C} \rightarrow [0, 1, 0, 0]$$
$$\text{G} \rightarrow [0, 0, 1, 0]$$
$$\text{T} \rightarrow [0, 0, 0, 1]$$

Agar humara DNA sequence **60 bases** lamba hai:
* Har base ko 4 numbers se represent kiya jayega.
* Ek sequence ki shape hogi: $60 \times 4$ matrix.
* Jab hum is 2D matrix ko **Flatten** karenge (jaise MNIST mein kiya tha): 
  $$\text{Input Features} = 60 \times 4 = 240 \text{ inputs!}$$

---

## 3. PyTorch Custom Dataset: Har Line Ka Matlab

PyTorch mein real-world data handle karne ke liye hum `torch.utils.data.Dataset` class ko inherit karke apna custom dataset banate hain. Iske 3 main functions hote hain jo humein likhne hote hain:

1. `__init__(self, ...)`: Yahan hum data ko load, generate ya initialize karte hain.
2. `__len__(self)`: Ye total number of samples return karta hai (e.g., total sequences).
3. `__getitem__(self, idx)`: Ye pure dataset se index `idx` wale ek single sample (sequence tensor, label tensor) ko return karta hai.

Chalo iska logic dekhte hain:

```python
class DNADataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # One-hot encode character sequence to a PyTorch Tensor
        encoded_seq = one_hot_encode(seq)
        
        # Tensor convert karein
        x = torch.tensor(encoded_seq, dtype=torch.float32)
        y = torch.tensor([label], dtype=torch.float32)
        
        return x, y
```

---

## 4. Model Architecture: DNA MLP Classifier

Humara model sequence ke flattened inputs lega ($60 \times 4 = 240$) aur predict karega ki kya ye promoter hai (0 ya 1):

```
DNA Sequence (Length 60)
         │
  [One-Hot Encode]
         │
   Matrix (60 x 4)
         │
     [Flatten]
         │
   Vector (240,)
         │
   [nn.Linear(240, 64)]  ──> ReLU ──> Dropout(0.2)
         │
   [nn.Linear(64, 32)]   ──> ReLU ──> Dropout(0.1)
         │
   [nn.Linear(32, 1)]    ──> Sigmoid (0 to 1 probability)
         │
   Binary Output (Promoter / Non-Promoter)
```

### PyTorch Code for Network:
```python
class DNAMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # 2D (60, 4) ko 1D (240) banayega
        
        self.network = nn.Sequential(
            nn.Linear(240, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Binary Classification ke liye Sigmoid!
        )

    def forward(self, x):
        # x is (batch_size, 60, 4)
        x = self.flatten(x)   # Shape: (batch_size, 240)
        out = self.network(x) # Shape: (batch_size, 1)
        return out
```

---

## 5. Training Step-by-Step

Kyunki ye **Binary Classification** hai (Promoter ya Non-Promoter), hum log-loss function use karenge:
`criterion = nn.BCELoss()`

Training loop ke steps:
1. `optimizer.zero_grad()`: Pichle steps ke gradient vectors clear karo.
2. `outputs = model(inputs)`: Batch inputs ko model ke forward pass se nikaal kar predictions calculate karo.
3. `loss = criterion(outputs, targets)`: Real labels aur predicted probabilities ke beech log loss nikalo.
4. `loss.backward()`: Backpropagation run karke har layer ke parameters ($W, b$) ka gradient dL/dW calculate karo.
5. `optimizer.step()`: Adam optimizer se learning rate aur momentum apply karke weights update karo.

---

## 6. Chalo Ab Apni File Banao Aur Run Karo!

Ab hum aage badhenge. Aap ko is step-by-step practice ko complete karna hai.

### Practice Task:
1. Ek new file create karo: [dna_promoter_project.py](file:///g:/plans/mml/ml_learning_plan/classification_lessons/practice%20coding/dna_promoter_project.py).
2. Isme main niche complete, comment-loaded code de raha hoon, ise paste karo.
3. Code ko deeply read karo, aur ise run karo!
