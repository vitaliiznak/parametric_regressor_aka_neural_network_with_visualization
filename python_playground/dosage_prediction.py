import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
def generate_immune_response_data(num_points: int) -> tuple[np.ndarray, np.ndarray]:
    D = np.linspace(0, 100, num_points)  # Dosage amount from 0 to 100 mg
    I = (D / 8) / ((D / 8)**2 + 2) + 0.4 * np.sin(0.45 * (D / 8))  # Immune response
    return D, I

# Define the neural network model
class ImmuneResponseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden1 = nn.Linear(1, 4)  # First hidden layer with 3 neurons
        self.hidden2 = nn.Linear(4, 2)  # Second hidden layer with 2 neurons
        self.output = nn.Linear(2, 1)   # Output layer
        self.activation = nn.Tanh()     # Activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        return self.output(x)

# Train the model
def train_model(
    model: nn.Module, 
    dataloader: DataLoader, 
    epochs: int, 
    learning_rate: float
) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(1, epochs + 1):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if epoch % (epochs // 10) == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")

# Plot the results
def plot_results(
    model: nn.Module, 
    D_train: np.ndarray, 
    I_train: np.ndarray, 
    scaler: StandardScaler
) -> None:
    model.eval()
    with torch.no_grad():
        D_vals = np.linspace(0, 100, 100).reshape(-1, 1)  # Extend to 100
        D_vals_normalized = scaler.transform(D_vals)
        D_vals_tensor = torch.FloatTensor(D_vals_normalized)
        I_preds = model(D_vals_tensor).numpy()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(D_train, I_train, label="Training Data", color='blue')
    plt.plot(D_vals, I_preds, label="Model Prediction", color='red')
    plt.xlabel("Antibiotic Dosage (mg)")
    plt.ylabel("Immune Response")
    plt.title("Immune Response vs. Antibiotic Dosage")
    plt.legend()
    plt.show()

def main() -> None:
    # Generate data
    num_data_points = 300
    D_data, I_data = generate_immune_response_data(num_data_points)
    
    # Normalize data
    scaler = StandardScaler()
    D_data_normalized = scaler.fit_transform(D_data.reshape(-1, 1))
    
    D_tensor = torch.FloatTensor(D_data_normalized)
    I_tensor = torch.FloatTensor(I_data).unsqueeze(1)

    # Create dataset and dataloader
    dataset = TensorDataset(D_tensor, I_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = ImmuneResponseModel()

    # Training parameters
    num_epochs = 300
    learning_rate = 0.01

    # Train the model
    train_model(model, dataloader, num_epochs, learning_rate)

    # Plot the results
    plot_results(model, D_data, I_data, scaler)

if __name__ == "__main__":
    main()