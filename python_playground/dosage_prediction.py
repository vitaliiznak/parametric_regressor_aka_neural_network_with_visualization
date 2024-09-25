import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
def generate_vaccine_data(num_points: int) -> tuple[np.ndarray, np.ndarray]:
    D = np.linspace(0, 10, num_points)  # Dosage amount
    E = D / (D**2 + 2) + np.sin(0.5 * D)  # Vaccine effectiveness
    return D, E

# Define the neural network model
class VaccineEffectivenessModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(1, 2)  # Hidden layer with 18 neurons
        self.output = nn.Linear(2, 1)  # Output layer


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.ReLU(self.hidden(x))
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
    E_train: np.ndarray, 
    scaler: StandardScaler
) -> None:
    model.eval()
    with torch.no_grad():
        D_vals = np.linspace(0, 10, 100).reshape(-1, 1)
        D_vals_normalized = scaler.transform(D_vals)
        D_vals_tensor = torch.FloatTensor(D_vals_normalized)
        E_preds = model(D_vals_tensor).numpy()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(D_train, E_train, label="Training Data", color='blue')
    plt.plot(D_vals, E_preds, label="Model Prediction", color='red')
    plt.xlabel("Dosage Amount")
    plt.ylabel("Vaccine Effectiveness")
    plt.title("Vaccine Effectiveness vs. Dosage Amount")
    plt.legend()
    plt.show()

def main() -> None:
    # Generate data
    num_data_points = 100
    D_data, E_data = generate_vaccine_data(num_data_points)
    
    # Normalize data
    scaler = StandardScaler()
    D_data_normalized = scaler.fit_transform(D_data.reshape(-1, 1))
    
    D_tensor = torch.FloatTensor(D_data_normalized)
    E_tensor = torch.FloatTensor(E_data).unsqueeze(1)

    # Create dataset and dataloader
    dataset = TensorDataset(D_tensor, E_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = VaccineEffectivenessModel()

    # Training parameters
    num_epochs = 5000
    learning_rate = 0.01

    # Train the model
    train_model(model, dataloader, num_epochs, learning_rate)

    # Plot the results
    plot_results(model, D_data, E_data, scaler)

if __name__ == "__main__":
    main()