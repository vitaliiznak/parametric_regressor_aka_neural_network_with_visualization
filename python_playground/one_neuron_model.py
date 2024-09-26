import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 1. Generate the dataset for the study time vs exam performance problem


def true_function(S):
    return (S / (10 + S**2)) + 0.3 * np.sin(0.4 * S)


# Generate study time (S) from 0 to 12 hours
S = np.linspace(0, 12, 100).reshape(-1, 1)  # 100 data points
P = true_function(S)

# Convert the data to PyTorch tensors
S_tensor = torch.Tensor(S)  # Input (study time in hours)
P_tensor = torch.Tensor(P)  # Output (exam performance)

# Define the Swish activation function


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 2. Define the model: One hidden neuron with Swish activation


class OneNeuronModel(nn.Module):
    def __init__(self):
        super(OneNeuronModel, self).__init__()
        # Hidden layer: One neuron
        self.fc1 = nn.Linear(1, 1)  # One input (S), one neuron in hidden layer

        # Output layer: One neuron for final prediction
        self.fc2 = nn.Linear(1, 1)  # One output (P)

        # Swish activation function
        self.swish = Swish()

    def forward(self, x):
        hidden_output = self.swish(self.fc1(x))  # Apply Swish activation
        return self.fc2(hidden_output)  # Output layer


# 3. Create the model, define loss function and optimizer
model = OneNeuronModel()
criterion = nn.MSELoss()  # Mean Squared Error Loss
# Stochastic Gradient Descent
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Train the model
epochs = 1000
for epoch in range(epochs):
    model.train()

    # Forward pass
    outputs = model(S_tensor)
    loss = criterion(outputs, P_tensor)

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
    predicted = model(S_tensor).numpy()  # Get predictions from the model

# Plot the true function and the model predictions
plt.figure(figsize=(8, 6))
plt.plot(S, P, 'b-', label='True function (Performance vs Study Time)')
plt.plot(S, predicted, 'r--', label='Model Prediction')
plt.xlabel('Study Time (hours)')
plt.ylabel('Exam Performance')
plt.title('Optimal Study Time vs Exam Performance with Swish Activation')
plt.legend()
plt.show()
