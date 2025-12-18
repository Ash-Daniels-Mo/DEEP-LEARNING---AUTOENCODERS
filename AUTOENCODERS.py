#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


# In[ ]:


transform = transforms.ToTensor()

# Training data: digits 0–8 only
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
train_indices = [i for i, (_, y) in enumerate(train_dataset) if y not in  [2, 4, 6, 8]]
train_loader = DataLoader(
    Subset(train_dataset, train_indices),
    batch_size=128,
    shuffle=True
)

# Test data: all digits
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# In[ ]:


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat.view(-1, 1, 28, 28)


# In[4]:


model = AutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 15
loss_history = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for x, _ in train_loader:
        x = x.to(device)

        x_hat = model(x)
        loss = criterion(x_hat, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")


# In[5]:


plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss")
plt.title("Training Loss Curve")
plt.show()


# In[7]:


model.eval()

# Store one example per digit
digit_examples = {}

with torch.no_grad():
    for x, y in test_loader:
        for img, label in zip(x, y):
            label = label.item()
            if label not in digit_examples:
                digit_examples[label] = img
            if len(digit_examples) == 10:
                break
        if len(digit_examples) == 10:
            break

# Stack images in digit order
inputs = torch.stack([digit_examples[d] for d in range(10)]).to(device)

# Reconstruct
with torch.no_grad():
    outputs = model(inputs)

# Plot
plt.figure(figsize=(15, 4))

for i in range(10):
    # Original
    plt.subplot(2, 10, i + 1)
    plt.imshow(inputs[i].cpu().squeeze(), cmap="gray")
    plt.title(f"{i}")
    plt.axis("off")

    # Reconstructed
    plt.subplot(2, 10, i + 11)
    plt.imshow(outputs[i].cpu().squeeze(), cmap="gray")
    plt.title("Recon")
    plt.axis("off")

plt.suptitle("Original vs Reconstructed Images for All Digits (0–9)")
plt.show()


# In[8]:


from collections import defaultdict

model.eval()

digit_errors = defaultdict(list)

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        x_hat = model(x)

        batch_errors = torch.mean((x - x_hat) ** 2, dim=(1, 2, 3))

        for err, label in zip(batch_errors.cpu().numpy(), y.numpy()):
            digit_errors[label].append(err)

# Compute mean error per digit
mean_digit_errors = {d: np.mean(errs) for d, errs in digit_errors.items()}

# Print results
print("Mean Reconstruction Error per Digit:")
for d in sorted(mean_digit_errors.keys()):
    print(f"Digit {d}: {mean_digit_errors[d]:.6f}")


# In[11]:


digits = list(mean_digit_errors.keys())
values = list(mean_digit_errors.values())

plt.figure(figsize=(8, 4))
plt.bar(digits, values)
plt.xlabel("Digit")
plt.ylabel("Mean Reconstruction Error")
plt.title("Reconstruction Error per Digit")
plt.show()


# In[10]:


errors = []
labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        x_hat = model(x)

        batch_error = torch.mean((x - x_hat) ** 2, dim=(1, 2, 3))
        errors.extend(batch_error.cpu().numpy())
        labels.extend(y.numpy())

errors = np.array(errors)
labels = np.array(labels)

normal_errors = errors[labels != 9]
anomaly_errors = errors[labels == 9]


# In[ ]:




