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
  inputSize: 3,
  layers: [4, 4, 1],
  activations: ['tanh', 'tanh', 'tanh']
});

// Hyperparameters
const learningRate = 0.01;
const epochs = 100;
const batchSize = 2;

// Training loop
for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = new Value(0);

    // Mini-batch training
    for (let i = 0; i < xs.length; i += batchSize) {
        const batchXs = xs.slice(i, i + batchSize);
        const batchYt = yt.slice(i, i + batchSize);

        const ypred = batchXs.map(x => n.forward(x.map(val => new Value(val))) as Value);

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

    // Log average loss for the epoch
    console.log(`Epoch ${epoch + 1}, Average Loss: ${totalLoss.data / xs.length}`);

    // Early stopping (optional)
    if (totalLoss.data / xs.length < 0.01) {
        console.log(`Converged at epoch ${epoch + 1}`);
        break;
    }
}

// Evaluation
function evaluate(x: number[]): number {
    const result = n.forward(x.map(val => new Value(val)));
    return (result as Value).data;
}

console.log("Evaluation:");
xs.forEach((x, i) => {
    console.log(`Input: [${x}], Predicted: ${evaluate(x).toFixed(4)}, Actual: ${yt[i]}`);
});

// Visualize loss function tree
const dotString = loss.toDot();
console.log("Loss function tree DOT representation:");
console.log(dotString);

// Render loss function tree visualization
const viz = new Viz();
viz.renderSVGElement(dotString)
    .then(element => {
        document.getElementById("graph").appendChild(element);
    })
    .catch(error => {
        console.error(error);
    });