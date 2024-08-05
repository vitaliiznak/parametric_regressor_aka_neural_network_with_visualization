import MLP from "./NeuralNetwork/MLP";
import Value from "./NeuralNetwork/Value";


function backwardAll(values: Value | Value[]): void {
  if (Array.isArray(values)) {
    values.forEach(v => v.backward());
  } else {
    values.backward();
  }
}

// Example usage:
// Example usage:
const mlp = new MLP(3, [4, 4, 1], ['ReLU', 'Tanh', 'Sigmoid']);
const x = [new Value(1.0), new Value(2.0), new Value(3.0)];
const y = mlp.forward(x);
console.log(y.toString());
backwardAll(y);
console.log(mlp.parameters());