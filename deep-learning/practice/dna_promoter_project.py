"""
DNA Promoter Classifier — Step-by-Step Bio-AI Project
======================================================

Ye project kya karega:
  - Synthetic DNA sequences (60 bp length) generate karega.
  - Kuch sequences mein 'TATAAT' (TATA Box) aur 'TTGACA' (-35 region) motifs honge. Ye Promoters (Class 1) hain.
  - Baki random sequences honge (Class 0).
  - DNA sequences ko One-Hot Encode karega (A, C, G, T -> vectors).
  - Custom PyTorch Dataset aur DataLoader banayega.
  - Ek MLP Neural Network train karega jo DNA pattern recognize karega!

HAR LINE KA MATLAB COMMENTS MEIN EXPLAINED HAI!
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# For reproducibility (har run pe same data aur results milein)
random.seed(42)
torch.manual_seed(42)

# ============================================================
# SECTION 1: DNA Data Generator (Synthetic Biology)
# ============================================================

def generate_random_dna(length):
    """Generates a random DNA string of a given length."""
    return "".join(random.choice(['A', 'C', 'G', 'T']) for _ in range(length))

def inject_motif(sequence, motif, position, mutation_rate=0.15):
    """
    Injects a motif into a sequence at a specific position.
    mutation_rate=0.15 means 15% chance that a letter in the motif mutates,
    which is realistic since biology has mutations!
    """
    seq_list = list(sequence)
    motif_list = list(motif)
    
    for i in range(len(motif)):
        # If random value is greater than mutation rate, keep the motif base.
        # Otherwise, mutate it to a random base!
        if random.random() > mutation_rate:
            seq_list[position + i] = motif_list[i]
        else:
            seq_list[position + i] = random.choice(['A', 'C', 'G', 'T'])
            
    return "".join(seq_list)

def generate_dataset(num_samples=4000):
    """
    Generates a balanced dataset of promoters (Class 1) and non-promoters (Class 0).
    Sequence length: 60 bp
    """
    sequences = []
    labels = []
    
    half_samples = num_samples // 2
    
    # 1. Promoter sequences (Class 1)
    for _ in range(half_samples):
        # Start with a random DNA sequence of length 60
        seq = generate_random_dna(60)
        # Inject TATA Box (TATAAT) at position 10
        seq = inject_motif(seq, "TATAAT", position=10)
        # Inject -35 region (TTGACA) at position 35
        seq = inject_motif(seq, "TTGACA", position=35)
        
        sequences.append(seq)
        labels.append(1)  # 1 = Promoter
        
    # 2. Non-Promoter sequences (Class 0)
    for _ in range(half_samples):
        # Simply random DNA sequence without specific motifs injected
        seq = generate_random_dna(60)
        sequences.append(seq)
        labels.append(0)  # 0 = Non-Promoter
        
    # Shuffle the dataset
    dataset = list(zip(sequences, labels))
    random.shuffle(dataset)
    
    shuffled_seqs, shuffled_labels = zip(*dataset)
    return list(shuffled_seqs), list(shuffled_labels)

print("Generating synthetic DNA sequences...")
sequences, labels = generate_dataset(num_samples=4000)
print(f"Dataset generated! Total samples: {len(sequences)}")
print(f"Sample Promoter (Label 1): {sequences[labels.index(1)]}")
print(f"Sample Non-Promoter (Label 0): {sequences[labels.index(0)]}\n")

# ============================================================
# SECTION 2: One-Hot Encoding DNA
# ============================================================

def one_hot_encode(sequence):
    """
    Converts a DNA string of length L into a 2D matrix of shape (L, 4).
    A -> [1, 0, 0, 0]
    C -> [0, 1, 0, 0]
    G -> [0, 0, 1, 0]
    T -> [0, 0, 0, 1]
    """
    mapping = {
        'A': [1.0, 0.0, 0.0, 0.0],
        'C': [0.0, 1.0, 0.0, 0.0],
        'G': [0.0, 0.0, 1.0, 0.0],
        'T': [0.0, 0.0, 0.0, 1.0]
    }
    
    # Default to all zeros if some invalid character appears
    encoded = [mapping.get(base, [0.0, 0.0, 0.0, 0.0]) for base in sequence]
    return encoded

# Test One-Hot Encoding
test_seq = "ACGT"
encoded_test = one_hot_encode(test_seq)
print(f"One-Hot encoding test: '{test_seq}' becomes:")
for char, vec in zip(test_seq, encoded_test):
    print(f"  {char} -> {vec}")
print()

# ============================================================
# SECTION 3: Custom PyTorch Dataset
# ============================================================

class DNADataset(Dataset):
    """
    Custom Dataset class for DNA sequences.
    This converts DNA string sequence to One-Hot encoded Tensor dynamically.
    """
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        # Total samples return karta hai
        return len(self.sequences)
        
    def __getitem__(self, idx):
        # Ek single sample uthata hai
        seq_str = self.sequences[idx]
        label_val = self.labels[idx]
        
        # One-hot encode
        encoded = one_hot_encode(seq_str)
        
        # PyTorch Tensors mein convert karo
        # Shape: (60, 4) — matrix of floats
        x = torch.tensor(encoded, dtype=torch.float32)
        # Shape: (1,) — target value float32 (for BCELoss)
        y = torch.tensor([label_val], dtype=torch.float32)
        
        return x, y

# Train/Test Split (80% training, 20% validation)
split_idx = int(len(sequences) * 0.8)

train_seqs, train_lbls = sequences[:split_idx], labels[:split_idx]
test_seqs, test_lbls = sequences[split_idx:], labels[split_idx:]

# Datasets banayein
train_dataset = DNADataset(train_seqs, train_lbls)
test_dataset = DNADataset(test_seqs, test_lbls)

# DataLoader banayein (Data batches mein supply karega)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training batches: {len(train_loader)} (32 samples each)")
print(f"Testing batches:  {len(test_loader)} (32 samples each)\n")

# ============================================================
# SECTION 4: Network Definition — DNA MLP Classifier
# ============================================================

class DNAMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten layer: (batch_size, 60, 4) -> (batch_size, 240)
        # Kyunki nn.Linear ko 1D vector chahiye inputs mein
        self.flatten = nn.Flatten()
        
        # Dense Network
        self.network = nn.Sequential(
            nn.Linear(240, 64),       # Input size = 240 (60 bases * 4 categories)
            nn.ReLU(),                # Non-linearity
            nn.Dropout(0.2),          # Regularization (avoids overfitting)
            nn.Linear(64, 32),        # Hidden layer 2
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),         # Output layer: 1 probability score
            nn.Sigmoid()              # Sigmoid maps output score to 0.0 - 1.0
        )
        
    def forward(self, x):
        # x shape: (batch_size, 60, 4)
        x = self.flatten(x)       # (batch_size, 240)
        out = self.network(x)     # (batch_size, 1)
        return out

model = DNAMLP()
print(model)

# Count parameters:
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {params:,}\n")

# ============================================================
# SECTION 5: Loss and Optimizer
# ============================================================

criterion = nn.BCELoss()  # Binary Cross Entropy (Binary Classification ke liye)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============================================================
# SECTION 6: Training Loop
# ============================================================

EPOCHS = 15

print("="*50)
print("TRAINING DNAMLP MODEL START!")
print("="*50)

for epoch in range(EPOCHS):
    model.train()  # Dropout ON
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)             # Shape: (batch_size, 1)
        loss = criterion(outputs, batch_y)   # Loss calculate karo
        
        # Backward pass + updates
        optimizer.zero_grad()                # Reset old gradients
        loss.backward()                      # Gradients nikaalein (dLoss/dW)
        optimizer.step()                     # Weights update karein
        
        # Statistics track karein
        running_loss += loss.item() * batch_X.size(0)
        
        # Predictions (agar probability >= 0.5 to class 1, nahi to 0)
        predictions = (outputs >= 0.5).float()
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    # Validation / Test Evaluation (Har epoch ke baad)
    model.eval()  # Dropout OFF
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():  # No gradients needed for evaluation (memory saving)
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_X.size(0)
            
            predictions = (outputs >= 0.5).float()
            test_correct += (predictions == batch_y).sum().item()
            test_total += batch_y.size(0)
            
    val_loss = test_loss / test_total
    val_acc = 100.0 * test_correct / test_total
    
    print(f"Epoch [{epoch+1:2d}/{EPOCHS:2d}] | "
          f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:5.1f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.1f}%")

print("="*50)
print("TRAINING COMPLETE!")
print(f"Final Validation Accuracy: {val_acc:.1f}%")
print("="*50)

# ============================================================
# SECTION 7: Inspecting Predictions
# ============================================================

# Chalo 5 random samples uthate hain test set se aur check karte hain:
model.eval()
print("\nTesting model on 5 sample sequences:")
with torch.no_grad():
    for i in range(5):
        seq_str = test_seqs[i]
        true_lbl = test_lbls[i]
        
        # Run through model
        encoded = one_hot_encode(seq_str)
        # Add batch dimension: (60, 4) -> (1, 60, 4)
        x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
        prob = model(x).item()
        pred_lbl = 1 if prob >= 0.5 else 0
        
        # Check if motifs are present in string to verify biology!
        has_tata = "TATAAT" in seq_str
        has_35 = "TTGACA" in seq_str
        
        status = "CORRECT" if pred_lbl == true_lbl else "WRONG"
        print(f"Sample #{i+1}:")
        print(f"  Sequence: {seq_str[:15]}...{seq_str[35:50]}...")
        print(f"  Contains TATAAT? {has_tata} | Contains TTGACA? {has_35}")
        print(f"  Model Prob: {prob:.4f} | Prediction: {pred_lbl} | Actual: {true_lbl} [{status}]")
        print("-" * 50)

# Save model weights
torch.save(model.state_dict(), 'dna_promoter_model.pt')
print("\nModel weights saved as 'dna_promoter_model.pt'")
