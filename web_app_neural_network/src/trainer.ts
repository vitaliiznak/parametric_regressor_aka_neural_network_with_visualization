import { MLP } from "./NeuralNetwork/mlp";
import { Value } from "./NeuralNetwork/value";
import { BackwardStepGradients, BackwardStepGradientsPerConnection, ForwardStepResults, TrainingConfig, TrainingStepResult } from "./types";

export class Trainer {
  _network: MLP;
  private currentIteration: number = 0;

  private xs: number[][] = [];
  private yt: number[] = [];
  private currentOutput: Value[] | null = null;
  private currentLoss: Value | null = null;
  private currentDataIndex: number = 0;
  private currentBatchInputs: number[][] = [];
  private currentBatchTargets: number[] = [];

  stepBackward: () => BackwardStepGradientsPerConnection[];

  constructor(network: MLP, config: TrainingConfig) {
    this._network = network.clone();

    this.stepBackward = this.stepBackwardAndGetGradientsGroupedByConnection
  }

  get network(): MLP {
    return this._network;
  }

  setTrainingData(xs: number[][], yt: number[]): void {
    this.xs = xs;
    this.yt = yt;
    this.currentIteration = 0;
  }

  shuffleDataset() {
    const shuffled = this.xs.map((x, i) => ({ x, y: this.yt[i] }));
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    this.xs = shuffled.map(item => item.x);
    this.yt = shuffled.map(item => item.y);
  }

  singleStepForward(): ForwardStepResults | null {
    if (this.currentDataIndex >= this.xs.length) {
      this.currentDataIndex = 0;
      return null;
    }

    const x = this.xs[this.currentDataIndex];
    const y = this.yt[this.currentDataIndex];
    this.currentOutput = this._network.forward(x.map(val => new Value(val)));

    this.currentBatchInputs.push(x);
    this.currentBatchTargets.push(y);

    const result: ForwardStepResults = {
      input: x,
      output: this.currentOutput.map(v => v.data),
    };

    this.currentDataIndex++;
    return result;
  }

  batchForwardStep(batchSize: number): ForwardStepResults[] {
    const results: ForwardStepResults[] = [];
    for (let i = 0; i < batchSize; i++) {
      const result = this.singleStepForward();
      if (result) {
        results.push(result);
      } else {
        break;
      }
    }
    return results;
  }

  calculateLoss(): Value | number | null {
    if (this.currentBatchInputs.length === 0) {
      console.error("No data in the current batch");
      return null;
    }

    this.currentLoss = this.calculateBatchLoss(this.currentBatchInputs, this.currentBatchTargets);

    console.log('Calculated loss:', this.currentLoss.data);

    // After calculating currentLoss
/*     console.log('Computation Tree for currentLoss:');
    if (this.currentLoss) {
      const compuationTreeString = this.currentLoss.printTree()
      console.log(compuationTreeString);

      // Derive batch size by counting 'add' operations
      const batchSize = this.deriveBatchSizeFromComputationTreeString(compuationTreeString);
      console.log(`Derived Batch Size From Computation Tree String: ${batchSize}`);
    } else {
      console.log('currentLoss is null');
    }
 */
    return this.currentLoss;
  }


  /**
 * Determines the batch size from a computational graph log.
 * 
 * The function primarily looks for the line starting with 'Value (^-1):'
 * and computes the batch size as the inverse of that value.
 * If such a line isn't found, it falls back to counting unique non-zero 'Value (Input):' entries.
 * 
 * @param computationalGraph - The computational graph log as a multi-line string.
 * @returns The determined batch size as a number.
 */
  private deriveBatchSizeFromComputationTreeString(computationalGraph: string): number {
    // Split the computational graph into lines
    const lines = computationalGraph.split('\n');

    // Regular expression to match 'Value (^-1): <number>'
    const inverseRegex = /^ *Value\s*\(\^-1\):\s*([0-9.]+)/;

    // Initialize batchSize as null
    let batchSize: number | null = null;

    // Iterate through each line to find 'Value (^-1):'
    for (const line of lines) {
      const match = line.match(inverseRegex);
      if (match) {
        const inverseValue = parseFloat(match[1]);
        if (inverseValue !== 0) {
          batchSize = 1 / inverseValue;
          break; // Assuming only one 'Value (^-1):' line is present
        }
      }
    }

    // If 'Value (^-1):' was found and batchSize determined
    if (batchSize !== null) {
      return batchSize;
    }

    // Fallback: Count unique non-zero 'Value (Input):' entries
    // Note: This is less reliable and may include parameters/activations
    const inputRegex = /^ *Value \(Input\):\s*([-+]?[0-9]*\.?[0-9]+)/;
    const inputValues = new Set<string>();

    for (const line of lines) {
      const match = line.match(inputRegex);
      if (match) {
        const inputValue = match[1].trim();
        // Ignore zero inputs or constants
        if (inputValue !== '0') {
          inputValues.add(inputValue);
        }
      }
    }

    return inputValues.size;
  }



  private calculateBatchLoss(inputs: number[][], targets: number[]): Value {
    let totalLoss = new Value(0);
    for (let i = 0; i < inputs.length; i++) {
      const pred = this._network.forward(inputs[i].map(val => new Value(val)))[0];
      const target = new Value(targets[i]);
      const diff = pred.sub(target);
      const squaredDiff = diff.mul(diff);
      console.log(`Batch ${i + 1}: Prediction = ${pred.data}, Target = ${target.data}, Squared Diff = ${squaredDiff.data}`);
      totalLoss = totalLoss.add(squaredDiff);
    }

    const batchSizeValue = new Value(inputs.length, []);
    const avgLoss = totalLoss.div(batchSizeValue);
    console.log(`Total Loss: ${totalLoss.data}, Avg Loss: ${avgLoss.data}`);
    return avgLoss;
  }

  stepBackwardAndGetGradientsGroupedByNeurons(): BackwardStepGradients | null {
    // Recalculate the loss before each backward step
    this.calculateLoss();

    if (!this.currentLoss) {
      console.error("Loss not calculated");
      return null;
    }

    console.log('Gradients before zeroing:', this._network.parameters().map(p => p.grad));
    this._network.zeroGrad();
    console.log('Gradients after zeroing:', this._network.parameters().map(p => p.grad));

    this.currentLoss.backward();

    const result: BackwardStepGradients = this._network.layers.flatMap((layer, layerIndex) =>
      layer.neurons.map((neuron, neuronIndex) => ({
        neuron: neuronIndex + 1,
        weights: neuron.w.length,
        bias: 1,
        gradients: [...neuron.w.map(w => w.grad), neuron.b.grad]
      }))
    );

    console.log('Gradients after backward pass:', result);

    return result;
  }

  stepBackwardAndGetGradientsGroupedByConnection(): BackwardStepGradientsPerConnection[] {
    // Recalculate the loss before each backward step
    this.calculateLoss();

    if (!this.currentLoss) {
      console.error("Loss not calculated");
      return [];
    }

    // Zero out existing gradients
    this._network.zeroGrad();

    // Perform backpropagation
    this.currentLoss.backward();

    const gradients: BackwardStepGradientsPerConnection[] = [];

    // Iterate through each layer and neuron to collect gradients
    this._network.layers.forEach((layer, layerIndex) => {
      layer.neurons.forEach((neuron, neuronIndex) => {
        neuron.w.forEach((weight, weightIndex) => {
          const fromNodeId = layerIndex === 0
            ? `neuron_-1_${weightIndex}` // Consistent with input node IDs
            : `neuron_${layerIndex - 1}_${weightIndex}`;
          const toNodeId = `neuron_${layerIndex}_${neuronIndex}`;
          const connectionId = `from_neuron_${layerIndex - 1}_${weightIndex}_to_neuron_${layerIndex}_${neuronIndex}`; // Standardized format

          gradients.push({
            connectionId,
            weightGradient: weight.grad,
            biasGradient: neuron.b.grad,
          });
        });
      });
    });

    console.log('Gradients after backward pass:', gradients);

    return gradients;
  }



  updateWeights(learningRate: number): TrainingStepResult {
    const oldWeights = this._network.parameters().map(p => p.data);
    this._network.parameters().forEach(p => {
      p.data -= learningRate * p.grad;
    });
    const newWeights = this._network.parameters().map(p => p.data);
    return {
      gradients: this._network.parameters().map(p => p.grad),
      oldWeights,
      newWeights
    };
  }

  getCurrentIteration(): number {
    return this.currentIteration;
  }

  resetBatch(): void {
    this.currentBatchInputs = [];
    this.currentBatchTargets = [];
    this.currentDataIndex = 0;
    console.log('Trainer batch has been reset.');
  }

}