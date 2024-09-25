# 💊 Dr. House’s Perfect Dosage Predictor - Project Requirements 💊

## 1. Objective

Develop an educational tool to illustrate how neural networks operate by modeling the relationship between medication dosage and its effectiveness.

- **Start with a single variable (dosage level) and plan for future expansion to include multiple features.**
- **Provide practical insights in a beginner-friendly manner.**
- **Enable manual control over the training process to deepen user understanding of neural network learning.**

## 2. Scenario

**"Dr. House’s Dosage Dilemma"**

Dr. House aims to optimize medication dosage to maximize effectiveness while minimizing side effects. Model the relationship between dosage percentage and effectiveness score using a neural network.

## 3. Key Details

- **Input (x):** Dosage level, scaled between 0 and 1.
- **Output (y):** Effectiveness score, scaled between 0 and 1.

### Underlying Function:

\[
y = \frac{\sin(2\pi x) + 0.5 \cdot \sin(4\pi x) + 1 + \text{noise}}{3}
\]

#### Noise:

Add random noise to simulate real-world data variability.

### Simplified Function Explanation:

**Two Sine Waves:**

- \(\sin(2\pi x)\): Creates a wave with a period of 1 unit, introducing the first wiggle.
- \(0.5 \cdot \sin(4\pi x)\): Creates a wave with a period of 0.5 units, adding a secondary wiggle.

**Constant Term:**

- \(+1\): Shifts the entire function upwards for better normalization.

**Scaling:**

- The entire expression is divided by 3 to scale \(y\) between 0 and 1.

## 4. Project Tasks

### A. Data Generation

**Generate Data Points:**

- Use the simplified function with two sine components to create two wiggles across the dosage range of 0-1.
- Incorporate random noise to mimic real-world variability.

**Function with Noise:**

\[
y = \frac{\sin(2\pi x) + 0.5 \cdot \sin(4\pi x) + 1 + \text{noise}}{3}
\]

**Data Size:**

- Create 1000 data points for robust training and testing.

**Implementation Example in TypeScript:**

```typescript
function generateData(numPoints: number): { x: number; y: number }[] {
  const data = [];
  for (let i = 0; i < numPoints; i++) {
    const x = Math.random(); // Dosage from 0 to 1
    const noise = (Math.random() - 0.5) * 0.05; // Noise between -0.025 and 0.025
    const y = (Math.sin(2 * Math.PI * x) 
            + 0.5 * Math.sin(4 * Math.PI * x) 
            + 1 
            + noise) / 3;
    data.push({ x, y });
  }
  return data;
}

const dataset = generateData(1000);
```

### B. Neural Network Implementation

**Architecture:**

- **Input Layer:** 1 node (dosage level).
- **Hidden Layers:** 2 layers with 10 nodes each.
- **Output Layer:** 1 node (predicted effectiveness score).

**Activation Functions:**

- Use non-linear activation functions (e.g., ReLU) in hidden layers to capture the nonlinear relationship.

**Training:**

- Implement gradient descent training from scratch in TypeScript without external libraries.

**Basic Neural Network Structure in TypeScript:**

```typescript
class NeuralNetwork {
  layers: number[];
  weights: number[][][];
  biases: number[][];

  constructor(layers: number[]) {
    this.layers = layers;
    this.weights = [];
    this.biases = [];
    // Initialize weights and biases
    for (let i = 0; i < layers.length - 1; i++) {
      const weightLayer = [];
      const biasLayer = [];
      for (let j = 0; j < layers[i + 1]; j++) {
        weightLayer.push(Array(layers[i]).fill(0).map(() => Math.random() - 0.5));
        biasLayer.push(Math.random() - 0.5);
      }
      this.weights.push(weightLayer);
      this.biases.push(biasLayer);
    }
  }

  // Implement forward pass, backpropagation, and weight updates here
}
```

### C. Manual Training Control

**Interactive Controls:**

- **Forward Step:** Perform a forward pass on one data point.
- **Calculate Loss:** Compute and display the loss after forward passes.
- **Backward Step:** Execute backpropagation to compute gradients.
- **Update Weights:** Manually update neural network weights based on gradients.
- **Reset:** Reset the training process to its initial state.

**Visualization Components:**

- **ForwardStepsVisualizer:** Display input-output pairs of each forward step.
- **LearningProcessVisualizer:** Show gradients and updated weights during backpropagation and weight updates.
- **TrainingStatus:** Display current iteration, loss value, and a progress bar.

### D. Visualization

**Initial 2D Plots:**

- Plot the true function versus the neural network’s predictions using 2D graphs.

**Interactivity:**

- Allow users to input their own dosage percentage and view the predicted effectiveness score in real-time.

### E. Code Explanation

**High-Level Overview:**

- Provide beginner-friendly explanations of neural network components, including layers, activation functions, and training processes.

**Conceptual Understanding:**

- Emphasize the intuition behind neural networks learning nonlinear relationships.

### F. Optimization and User Interaction

**Optimal Usage Identification:**

- Use the trained model to determine the optimal dosage level for maximum effectiveness.

**User Input Functionality:**

- Enable users to input dosage percentages and receive predicted effectiveness scores.

**Practical Recommendations:**

- Offer insights and recommendations based on model predictions.

### G. Expansion to Multiple Features

**Additional Features Discussion:**

- Explore expanding the model to include features such as patient age, weight, and health conditions.

**Complexity and Capability Implications:**

- Discuss how adding features increases model complexity and enhances predictive capabilities.

### H. Timeframe and Deliverables

**Timeframe:**

- Complete the project within 2 weeks.

**Deliverables:**

- **TypeScript Code:**
  - Data generation.
  - Neural network implementation.
  - Visualization components.
- **Interactive Training Controls:**
  - Manual buttons for training steps.
- **Documentation:**
  - In-code explanations for each step.
- **Interactive Visualizations:**
  - Compare true function with neural network predictions.
- **Analysis and Recommendations:**
  - Optimal dosage levels based on model.
- **Educational Content:**
  - Beginner-friendly explanations of neural network concepts.

## 5. Gradient Descent and Parameter Updates

### Objective

Explain the gradient descent algorithm and its role in updating neural network parameters.

### Content Requirements

**Gradients:**

- Define gradients as the rate of change of the loss function relative to weights and biases.

**Backpropagation:**

- Explain the backpropagation algorithm and its use of the chain rule to compute gradients.

**Gradient Descent Process:**

- Detail how gradients are utilized in gradient descent to adjust weights and biases to minimize loss.

**Interactive Visualizations:**

- Illustrate how parameter updates influence network predictions over iterations.

**Numerical Examples:**

- Provide examples of weight updates with actual numerical values to demonstrate parameter changes during training.

### Activities

**Code Snippets:**

- Implement backpropagation and parameter updates in TypeScript.

**Exercises:**

- Enable users to manually compute gradients and perform parameter updates on a simple network.

**Quizzes:**

- Assess understanding of gradient descent and backpropagation concepts.

### Resources

**Diagrams and Flowcharts:**

- Visual representations of gradient flow through the network.

**External Links:**

- Provide links to additional readings on optimization algorithms and neural network training.

## 6. UI/UX Implementation

### Frontend Technologies

- **Framework:** Solid.js
- **Styling:** Emotion/CSS

### Design Principles

- **Minimalistic and Intuitive:**
  - Clean UI with clear and concise information display.
- **Compact Controls:**
  - Small-sized buttons and controls to maximize screen space for visualizations.
- **Dynamic Batch Size:**
  - Batch size equals the number of forward steps performed by the user.
  - No batch training: Training occurs step-by-step based on user interaction.

### UI Components

**Buttons:**

- **Forward Step**
- **Calculate Loss**
- **Backward Step**
- **Update Weights**
- **Reset**

**Visualizers:**

- **ForwardStepsVisualizer:**
  - Displays input-output pairs.
- **LearningProcessVisualizer:**
  - Shows gradients and weight updates.

**TrainingStatus:**

- Displays current iteration, loss value, and a progress bar.

**Interactive Plot:**

- Allows users to input dosage percentage and view predicted effectiveness.

### Sample UI Layout

```
---------------------------------------------------------
| Dr. House’s Perfect Dosage Predictor                  |
|-------------------------------------------------------|
| [Forward Step] [Calculate Loss] [Backward Step]       |
| [Update Weights] [Reset]                              |
|-------------------------------------------------------|
| ForwardStepsVisualizer        | LearningProcessVisualizer |
| [Input: x] [Output: y]         | [Gradients] [Weights]    |
|                               |                          |
|-------------------------------------------------------|
| TrainingStatus: Iteration 10 | Loss: 0.123               |
| [Progress Bar]                                       |
|-------------------------------------------------------|
| Interactive Plot:                                     |
| [Input Dosage (0-1)]: [_____] [Predict]               |
| [Plot showing True Function vs Predictions]          |
---------------------------------------------------------
```

## 7. Final Outcome

- **Accurate Predictions:**
  - The neural network accurately predicts medication effectiveness based on dosage levels.
- **Optimal Dosage Identification:**
  - The tool identifies the optimal dosage that maximizes effectiveness while minimizing side effects.
- **Educational Value:**
  - Users gain a hands-on understanding of how neural networks learn and optimize complex, nonlinear relationships through interactive controls and visualizations.
- **Enhanced Learning Experience:**
  - Minimalistic and intuitive UI/UX design facilitates focused learning without unnecessary complexity.

### Implementation Tips

**Data Normalization:**

- **Input (x):** Ensure dosage levels are scaled between 0 and 1.
- **Output (y):** Scale effectiveness scores between 0 and 1 using the provided formula.

**Activation Functions:**

- Use activation functions like ReLU (Rectified Linear Unit) in hidden layers to introduce non-linearity.

**Loss Function:**

- Implement Mean Squared Error (MSE) as the loss function to measure the difference between predicted and actual effectiveness scores.

**Learning Rate:**

- Choose an appropriate learning rate (e.g., 0.01) to ensure stable and efficient training.

**Visualization Libraries:**

- Utilize simple charting libraries compatible with Solid.js for plotting graphs (e.g., Chart.js with Solid.js wrappers).

**State Management:**

- Manage the neural network’s state (weights, biases, gradients) efficiently using Solid.js’s reactive primitives.

