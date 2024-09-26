import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Define the immune response function


def immune_response_function(D_scaled: np.ndarray) -> np.ndarray:
    """
    Computes the immune response based on the scaled dosage.

    Args:
        D_scaled (np.ndarray): Scaled dosage values.

    Returns:
        np.ndarray: Immune response values.
    """
    D_scaled = D_scaled * 100
    term1 = (D_scaled / 8) / ((D_scaled / 8)**2 + 2)
    term2 = 0.4 * np.sin(0.45 * (D_scaled / 8))
    I = term1 + term2
    return I

# Generate synthetic data


def generate_immune_response_data(num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic immune response data based on dosage.

    Args:
        num_points (int): Number of data points to generate.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing normalized dosage amounts and normalized immune responses.
    """
    D = np.linspace(0, 1, num_points)  # Dosage amount from 0 to 1

    # Compute the immune response using the defined function
    I = immune_response_function(D)

    # Normalize to bring the function output between (-1, 1)
    I_min = np.min(I)
    I_max = np.max(I)
    I_normalized = 2 * (I - I_min) / (I_max - I_min) - \
        1  # Normalize to (-1, 1)

    return D, I_normalized

# Define the neural network model


class ImmuneResponseModel(nn.Module):
    """
    Neural network model for predicting immune response based on antibiotic dosage.
    """

    def __init__(self) -> None:
        super().__init__()
        self.hidden1 = nn.Linear(1, 4)  # First hidden layer with 4 neurons
        self.hidden2 = nn.Linear(4, 2)  # Second hidden layer with 2 neurons
        self.output = nn.Linear(2, 1)    # Output layer
        self.activation = nn.Tanh()      # Activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        return self.output(x)

# Train the model


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    learning_rate: float
) -> float:
    """
    Trains the neural network model.

    Args:
        model (nn.Module): The neural network model to train.
        dataloader (DataLoader): DataLoader for training data.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        float: Final training loss.
    """
    criterion = nn.MSELoss()
    final_loss = 0.0  # Initialize final loss

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0  # Cumulative loss for the epoch
        for inputs, targets in dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            model.zero_grad()
            loss.backward()

            # Manual update of parameters (Simple Gradient Descent)
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(dataloader)
        if epoch % max(epochs // 10, 1) == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}, Loss: {average_loss:.6f}")
            final_loss = average_loss  # Update final_loss with the latest average loss
    return final_loss  # Return the final loss

# Plot the results


def plot_results(
    model: nn.Module,
    D_train: np.ndarray,
    I_train: np.ndarray,
    scaler: StandardScaler,
    final_loss: float  # Added parameter for final loss
) -> None:
    """
    Plots the training data and model predictions, including the final loss.

    Args:
        model (nn.Module): Trained neural network model.
        D_train (np.ndarray): Training dosage data.
        I_train (np.ndarray): Training immune response data.
        scaler (StandardScaler): Scaler used for normalizing dosage data.
        final_loss (float): Final training loss to display on the plot.
    """
    model.eval()
    with torch.no_grad():
        # Generate dosage values scaled between 0 and 1
        # Changed from 0 to 100 to 0 to 1
        D_vals = np.linspace(0, 1, 100).reshape(-1, 1)
        D_vals_normalized = scaler.transform(D_vals)
        D_vals_tensor = torch.FloatTensor(D_vals_normalized)
        I_preds = model(D_vals_tensor).numpy()

    plt.figure(figsize=(10, 6))
    plt.scatter(D_train, I_train, label="Training Data", color='blue')
    plt.plot(D_vals, I_preds, label="Model Prediction", color='red')
    plt.xlabel("Antibiotic Dosage (scaled)")
    plt.ylabel("Immune Response")
    plt.title("Immune Response vs. Antibiotic Dosage")
    plt.legend()
    # Add final loss text to the plot
    plt.text(0.05, 0.95, f"Final Loss: {final_loss:.6f}",
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.xlim(0, 1)  # Set x-axis limits to 0-1
    plt.show()


def main() -> None:
    """
    Main function to generate data, train the model, and plot the results.
    """
    # Generate data
    num_data_points = 300
    D_data, I_data = generate_immune_response_data(num_data_points)

    # Normalize dosage data between 0 and 1 using StandardScaler
    scaler = StandardScaler()
    D_data_normalized = scaler.fit_transform(D_data.reshape(-1, 1))

    D_tensor = torch.FloatTensor(D_data_normalized)
    I_tensor = torch.FloatTensor(I_data).unsqueeze(1)

    # Create dataset and dataloader
    dataset = TensorDataset(D_tensor, I_tensor)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

    # Initialize model
    model = ImmuneResponseModel()

    # Training parameters
    num_epochs = 10
    learning_rate = 0.02

    # Train the model and capture final loss
    final_loss = train_model(model, dataloader, num_epochs, learning_rate)

    # Plot the results with final loss
    plot_results(model, D_data, I_data, scaler, final_loss)


if __name__ == "__main__":
    main()
