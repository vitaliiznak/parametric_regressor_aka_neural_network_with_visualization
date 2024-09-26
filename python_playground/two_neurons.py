import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# 1. Generate the dataset for the rocket fuel efficiency problem
def true_function(T):
    return T / (1 + 0.1 * T)

# Generate data
np.random.seed(42)  # for reproducibility
T = np.linspace(0, 100, 200).reshape(-1, 1)  # Thrust from 0 to 100 units, 200 points
E = true_function(T)

# Add some noise to make it more realistic
#noise = np.random.normal(0, 0.5, E.shape)
#E += noise

# Convert the data to PyTorch tensors
T_tensor = torch.Tensor(T)  # Input (thrust)
E_tensor = torch.Tensor(E)  # Output (fuel efficiency)

# Create TensorDataset and DataLoader
dataset = TensorDataset(T_tensor, E_tensor)
batch_size = 80
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 2. Define the model: Two hidden neurons
class TwoNeuronModel(nn.Module):
    def __init__(self):
        super(TwoNeuronModel, self).__init__()
        # Input layer has 1 input (thrust)
        self.fc1 = nn.Linear(1, 2)  # Two hidden neurons in the hidden layer
        self.fc2 = nn.Linear(2, 1)  # One output (fuel efficiency)
        self.sigmoid = nn.Tanh()  # Activation function for non-linearity
    
    def forward(self, x):
        hidden_output = self.sigmoid(self.fc1(x))  # Apply sigmoid activation to hidden neurons
        return self.fc2(hidden_output)  # Output layer

# 3. Create the model, define loss function and optimizer
model = TwoNeuronModel()
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.005)  # Stochastic Gradient Descent

# 4. Train the model
epochs = 3000  # Increased epochs due to smaller batch size
for epoch in range(epochs):
    model.train()
    
    for batch_T, batch_E in dataloader:
        # Forward pass
        outputs = model(batch_T)
        loss = criterion(outputs, batch_E)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 5. Test the model and plot the results
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    predicted = model(T_tensor).numpy()  # Get predictions from the model

# Plot the true function and the model predictions
plt.figure(figsize=(8, 6))
plt.plot(T, E, 'b-', label='True function (Efficiency vs Thrust)')
plt.plot(T, predicted, 'r--', label='Model Prediction')
plt.xlabel('Thrust')
plt.ylabel('Fuel Efficiency')
plt.title('Rocket Fuel Efficiency Prediction with Two Hidden Neurons (Simplified)')
plt.legend()
plt.show()
