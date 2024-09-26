import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 1. Generate the dataset (house area vs price) with noise
def true_function(A):
    return 2000 + 300 * A  # Linear relationship: Price = 2000 + 300 * Area

# Generate house area (A) from 50 to 200 square meters
A = np.linspace(50, 200, 100).reshape(-1, 1)  # 100 data points
P = true_function(A)

# Add random noise to the price
np.random.seed(42)  # For reproducibility
noise = np.random.normal(0, 500, P.shape)  # Mean 0, standard deviation 1000
P_noisy = P + noise

# Convert the data to PyTorch tensors
A_tensor = torch.Tensor(A)  # Input (house area in square meters)
P_tensor = torch.Tensor(P_noisy)  # Output (house price with noise)

# 2. Define the model: One neuron with no activation (linear relationship)
class OneNeuronLinearModel(nn.Module):
    def __init__(self):
        super(OneNeuronLinearModel, self).__init__()
        self.fc1 = nn.Linear(1, 1)  # One input (A), one output (P)
    
    def forward(self, x):
        return self.fc1(x)  # Linear layer

# 3. Create the model, define loss function and optimizer
model = OneNeuronLinearModel()
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.0001)  # Reduced learning rate

# 4. Train the model
epochs = 20000  # Further increased number of epochs
for epoch in range(epochs):
    model.train()
    
    # Forward pass
    outputs = model(A_tensor)
    loss = criterion(outputs, P_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print loss every 5000 epochs
    if (epoch + 1) % 5000 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 5. Get model parameters
weight = model.fc1.weight.item()
bias = model.fc1.bias.item()
print(f"Learned parameters - Weight: {weight:.4f}, Bias: {bias:.4f}")

# 6. Plot the results
plt.figure(figsize=(12, 8))
plt.scatter(A, P_noisy, color='g', alpha=0.5, label='Noisy data points')
plt.plot(A, P, 'b-', label='True function (Price vs Area)', linewidth=2)

# Plot the model's prediction using the learned parameters
A_range = np.array([A.min(), A.max()])
P_pred = weight * A_range + bias
plt.plot(A_range, P_pred, 'r--', label='Model Prediction', linewidth=2)

plt.xlabel('House Area (square meters)')
plt.ylabel('Price ($)')
plt.title('House Price Prediction with One Neuron Linear Model (Noisy Data)')
plt.legend()
plt.grid(True)
plt.show()

# Print model parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.data.numpy().flatten()}")