import torch 
import torch.nn as nn 

x=torch.tensor([0.15,0.08])
y= torch.tensor([1.0])
model =nn.Sequential(
    nn.Linear(2,4),
    nn.ReLU(),
    nn.Linear(4,1),
    nn.Sigmoid()
)
criterion=nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- Training Loop ---
print("Initial prediction before training:", model(x).item())

for step in range(50):
    output = model(x)             # Forward pass (prediction)
    loss = criterion(output, y)   # Calculate loss (BCELoss)
    
    optimizer.zero_grad()         # Reset gradients to zero
    loss.backward()               # Backpropagation (computes gradients)
    optimizer.step()              # Update weights (W and b)
    
    if step % 10 == 0:
        print(f"Step {step}: Prediction = {output.item():.4f}, Loss = {loss.item():.4f}")

print("Final prediction after training:", model(x).item())
