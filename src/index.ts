import { MLP } from "./NeuralNetwork/mlp";
import { Value } from "./NeuralNetwork/value";

// Example usage
const xs: number[][] = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
];
const yt: number[] = [1.0, -1.0, -1.0, 1.0];

// Create MLP
const n = new MLP({
  inputSize: 1,
  layers: [4, 4, 1],
  activations: ['tanh', 'tanh', 'tanh']
});

// Hyperparameters
const learningRate = 0.01;
const iterations = 100;
const batchSize = 2;

// Training loop
for (let iteration = 0; iteration < iterations; iteration++) {
    let totalLoss = new Value(0);

    // Mini-batch training
    for (let i = 0; i < xs.length; i += batchSize) {
        const batchXs = xs.slice(i, i + batchSize);
        const batchYt = yt.slice(i, i + batchSize);

        const ypred = batchXs.map(x => n.forward(x.map(val => new Value(val)))[0]);

        const loss = ypred.reduce((sum, ypred_el, j) => {
            const target = new Value(batchYt[j]);
            const diff = ypred_el.sub(target);
            const squaredError = diff.mul(diff);
            return sum.add(squaredError);
        }, new Value(0));

        // Accumulate total loss
        totalLoss = totalLoss.add(loss);

        // Backward pass
        n.zeroGrad();
        loss.backward();

        // Update parameters
        n.parameters().forEach(p => {
            p.data -= learningRate * p.grad;
        });

        // Inside the training loop, after calculating the loss
        console.log("Loss function tree:");
        console.log(loss.toDot());
    }

    // Log average loss for the iteration
    console.log(`Iteration ${iteration + 1}, Average Loss: ${totalLoss.data / xs.length}`);

    // Early stopping (optional)
    if (totalLoss.data / xs.length < 0.01) {
        console.log(`Converged at iteration ${iteration + 1}`);
        break;
    }
}

// Evaluation
function evaluate(x: number[]): number {
    const result = n.forward(x.map(val => new Value(val)));
    return result[0].data;
}

console.log("Evaluation:");
xs.forEach((x, i) => {
    console.log(`Input: [${x}], Predicted: ${evaluate(x).toFixed(4)}, Actual: ${yt[i]}`);
});



