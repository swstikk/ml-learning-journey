"""
MNIST Digit Classifier — Step by Step PyTorch Project
=====================================================

Ye project kya karega:
  - Haath se likhe hue digits ki images lega (0-9)
  - Neural Network un images ko dekhke batayega ki kaunsa digit hai
  - 97%+ accuracy ayegi!

HAR LINE DEEPLY EXPLAINED HAI. DHIRE PADH. RUSH MAT KAR.
"""

# ============================================================
# SECTION 1: Imports — Kya Chahiye?
# ============================================================

import os                 # Paths handle karne ke liye
import torch              # PyTorch — tensors + autograd
import torch.nn as nn     # Neural network layers (Linear, ReLU, etc.)
import torch.optim as optim  # Optimizers (Adam, SGD)

# torchvision = PyTorch ki image library
# Isme famous datasets hain (MNIST, CIFAR-10, ImageNet)
from torchvision import datasets, transforms

# DataLoader = Data ko batches mein todta hai
from torch.utils.data import DataLoader

# Matplotlib = Graphs aur images dikhane ke liye
import matplotlib.pyplot as plt

print("Step 1: Imports done!")

# ============================================================
# SECTION 2: Data Samjho — MNIST Kya Hai?
# ============================================================
"""
MNIST = 70,000 handwritten digit images (0-9)
  - 60,000 training images
  - 10,000 test images

Har image:
  - 28 x 28 pixels = 784 pixels total
  - Grayscale (black & white) — har pixel 0 to 255
  - 0 = black (background), 255 = white (ink)

Ek image kaisi dikhti hai:

  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 255 255 255 0 0 0 0 0 0 0 0  ← ink!
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 255 255 255 0 0 0 0 0 0 0 0 0
  ...ye milke digit "7" banata hai!

Label = 7 (ye image ka correct answer hai)
"""

# ============================================================
# SECTION 3: Data Download + Transform
# ============================================================

# Transform = Data ko model ke liye tayyar karo
# transforms.ToTensor():
#   - Image (0-255) ko Tensor (0.0-1.0) mein convert karta hai
#   - 255 se divide karta hai (normalization)
#   - Shape: (1, 28, 28) → 1 channel (grayscale), 28x28 pixels
#
# transforms.Normalize((0.1307,), (0.3081,)):
#   - MNIST dataset ka mean = 0.1307, std = 0.3081
#   - (pixel - mean) / std → centered around 0
#   - Training faster aur better hota hai centered data pe!

transform = transforms.Compose([
    transforms.ToTensor(),                        # 0-255 → 0.0-1.0
    transforms.Normalize((0.1307,), (0.3081,))    # Center around 0
])

# Download MNIST (pehli baar download hoga, ~11MB)
train_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
train_data = datasets.MNIST(
    root=train_dir,         # Kahan save kare (absolute path relative to script)
    train=True,             # Training set (60,000 images)
    download=True,          # Download kar agar nahi hai
    transform=transform     # Transform lagao
)

test_data = datasets.MNIST(
    root=train_dir,
    train=False,            # Test set (10,000 images)
    download=True,
    transform=transform
)

print(f"Training images: {len(train_data)}")   # 60,000
print(f"Test images:     {len(test_data)}")     # 10,000

# ============================================================
# SECTION 4: Ek Image Dekho — Kya Aata Hai?
# ============================================================

# Ek sample nikalo:
image, label = train_data[0]

print(f"\nImage shape: {image.shape}")   # torch.Size([1, 28, 28])
#                                           1  = channels (grayscale = 1)
#                                           28 = height
#                                           28 = width

print(f"Label: {label}")                  # e.g., 5 (ye digit "5" ki image hai)

print(f"Pixel values range: {image.min():.2f} to {image.max():.2f}")
# After normalize: roughly -0.4 to 2.8

# Image dikhao:
plt.figure(figsize=(4, 4))
plt.imshow(image.squeeze(), cmap='gray')   # squeeze: (1,28,28) → (28,28)
plt.title(f"Label: {label}")
plt.axis('off')
plt.savefig('mnist_sample.png', dpi=100, bbox_inches='tight')
plt.close()
print("Sample image saved as 'mnist_sample.png' — dekho!")

# ============================================================
# SECTION 5: Data ko Batches Mein Todo
# ============================================================

# DataLoader = Automatically batches banata hai
#
# batch_size=64:  64 images ek saath process karo
# shuffle=True:   Har epoch mein order random karo (overfitting se bachao)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)
#                                                    ^^^^^^^^^^^^^
#                                                    Test mein shuffle NAHI!
#                                                    Hamesha same order se test karo.

# Ek batch dekho:
batch_images, batch_labels = next(iter(train_loader))
print(f"\nBatch images shape: {batch_images.shape}")   # (64, 1, 28, 28)
#                                                          ^   ^  ^   ^
#                                                          |   |  |   |
#                                                     batch  chan H   W
#                                                     size   nel

print(f"Batch labels shape: {batch_labels.shape}")     # (64,)
print(f"Batch labels (first 10): {batch_labels[:10]}")

# ============================================================
# SECTION 6: Network Banao — Digit Classifier!
# ============================================================

"""
PROBLEM:
  Input:  Image of shape (1, 28, 28)   = 784 numbers
  Output: 10 probabilities (digit 0, 1, 2, ... 9)
  
  Kaunsa digit? → Jis class ki probability SABSE ZYADA!

ARCHITECTURE:
  Input (784) → Hidden1 (128) → Hidden2 (64) → Output (10)
  
  Activations: ReLU (hidden), NO sigmoid at output!
  
  Q: "Output pe Sigmoid kyun nahi?"
  A: Kyunki ye MULTI-CLASS hai (10 classes), not binary (2 classes)!
     Multi-class ke liye SOFTMAX chahiye.
     But hum BCELoss nahi use kar rahe — CrossEntropyLoss use karenge,
     jo KHUD softmax apply karta hai! (aage samjhayenge)
"""

class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # 784 → 128 → 64 → 10
        self.flatten = nn.Flatten()           # (1,28,28) → (784,)
        self.fc1     = nn.Linear(784, 128)    # First hidden layer
        self.fc2     = nn.Linear(128, 64)     # Second hidden layer
        self.fc3     = nn.Linear(64, 10)      # Output: 10 classes
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.2)        # 20% neurons off

    def forward(self, x):
        # x shape: (batch, 1, 28, 28)

        x = self.flatten(x)      # (batch, 784) ← 2D image → 1D vector!
        # Kyun flatten? Kyunki nn.Linear ko 1D input chahiye!
        # 28×28 = 784 numbers ki ek lambi list ban gayi.

        x = self.fc1(x)          # (batch, 128) ← W1 @ x + b1
        x = self.relu(x)         # ReLU activation
        x = self.dropout(x)      # Regularization

        x = self.fc2(x)          # (batch, 64) ← W2 @ x + b2
        x = self.relu(x)         # ReLU
        x = self.dropout(x)      # Regularization

        x = self.fc3(x)          # (batch, 10) ← W3 @ x + b3
        # NOTE: No activation here!
        # CrossEntropyLoss khud softmax lagayega!

        return x

model = DigitClassifier()

# Model ki summary:
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel created!")
print(f"Total parameters: {total_params:,}")
# 784*128 + 128 + 128*64 + 64 + 64*10 + 10 = 109,386 parameters!

# ============================================================
# SECTION 7: Loss Function — CrossEntropyLoss
# ============================================================

"""
Binary Classification (trading mein):
  Output: 1 number (probability of WIN)
  Loss:   BCELoss = -[y*log(p) + (1-y)*log(1-p)]

Multi-Class Classification (MNIST — 10 digits):
  Output: 10 numbers (raw scores, ek per class)
  Loss:   CrossEntropyLoss = -log(softmax(correct_class_score))

CrossEntropyLoss INTERNALLY kya karta hai:
  1. Raw scores pe SOFTMAX lagata hai → 10 probabilities (sum=1)
  2. Correct class ki probability pe -log lagata hai → Loss

Example:
  Model output: [1.2, 0.3, 3.5, 0.1, -0.5, 0.8, 0.2, -1.0, 0.4, 0.1]
                  0    1    2    3     4    5    6     7    8    9
  
  Actual label: 2
  
  Softmax:      [0.04, 0.02, 0.42, 0.01, 0.01, 0.03, 0.02, 0.01, 0.02, 0.01]
                                    ^^^^
                                    Class 2 ki probability = 0.42
  
  Loss = -log(0.42) = 0.868
  
  Agar model confident hota (class 2 = 0.99):
  Loss = -log(0.99) = 0.01  ← bahut kam!
"""

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============================================================
# SECTION 8: Training Loop — Pehli Baar Real Scale!
# ============================================================

print("\n" + "="*50)
print("TRAINING START!")
print("="*50)

EPOCHS = 10   # 10 baar pura dataset dekhega

for epoch in range(EPOCHS):
    model.train()                 # Dropout ON, training mode
    running_loss = 0.0            # Is epoch ka total loss track karo
    correct = 0                   # Kitne sahi predict kiye
    total = 0                     # Kitne total dekhe

    for batch_idx, (images, labels) in enumerate(train_loader):
        # images: (64, 1, 28, 28) — 64 images ka batch
        # labels: (64,) — 64 correct answers (0-9)

        # Forward pass:
        outputs = model(images)           # (64, 10) — 10 scores per image

        # Loss:
        loss = criterion(outputs, labels) # CrossEntropyLoss

        # Backward + Update:
        optimizer.zero_grad()             # Reset gradients
        loss.backward()                   # Backprop (automatic!)
        optimizer.step()                  # Adam update

        # Track progress:
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)   # Sabse bada score = predicted class
        #  ^                            ^
        #  |                            |
        # max value               along dimension 1 (classes)
        # (don't need)            returns index of max = predicted digit!

        total += labels.size(0)              # 64 samples counted
        correct += (predicted == labels).sum().item()  # kitne sahi the

    # Epoch complete! Stats print karo:
    train_acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    # ---- TEST accuracy (har epoch ke baad) ----
    model.eval()          # Dropout OFF, eval mode
    test_correct = 0
    test_total = 0

    with torch.no_grad():    # Gradients mat nikalo (speed + memory save)
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = 100 * test_correct / test_total

    print(f"Epoch [{epoch+1}/{EPOCHS}]  "
          f"Loss: {avg_loss:.4f}  "
          f"Train Acc: {train_acc:.1f}%  "
          f"Test Acc: {test_acc:.1f}%")

print("\n" + "="*50)
print(f"TRAINING COMPLETE!")
print(f"Final Test Accuracy: {test_acc:.1f}%")
print("="*50)

# ============================================================
# SECTION 9: Predictions Dekho — Sahi Ya Galat?
# ============================================================

# 10 random test images pe predict karo:
model.eval()
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

with torch.no_grad():
    for i, ax in enumerate(axes.flat):
        image, label = test_data[i * 100]   # Har 100th image lo
        output = model(image.unsqueeze(0))  # Add batch dimension: (1,1,28,28)
        _, predicted = torch.max(output, 1)
        pred = predicted.item()

        # Image dikhao:
        ax.imshow(image.squeeze(), cmap='gray')
        color = 'green' if pred == label else 'red'
        ax.set_title(f"Pred: {pred}\nActual: {label}",
                     color=color, fontsize=10)
        ax.axis('off')

plt.tight_layout()
plt.savefig('mnist_predictions.png', dpi=120, bbox_inches='tight')
plt.close()
print("\nPredictions saved as 'mnist_predictions.png' — dekho!")

# ============================================================
# SECTION 10: Model Save Karo
# ============================================================

torch.save(model.state_dict(), 'digit_classifier.pt')
print("Model saved as 'digit_classifier.pt'")

# Load kaise karo (future mein):
# model2 = DigitClassifier()
# model2.load_state_dict(torch.load('digit_classifier.pt'))
# model2.eval()
