import { MLP } from "./NeuralNetwork/mlp";
import { Value } from "./NeuralNetwork/value";
import { BackwardStepGradients, BackwardStepGradientsPerConnecrion, Prediction, TrainingConfig, TrainingStepResult } from "./types";

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
  
  stepBackward: () => BackwardStepGradientsPerConnecrion[];

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



  singleStepForward(): Prediction | null {
    if (this.currentDataIndex >= this.xs.length) {
      this.currentDataIndex = 0;
      return null;
    }

    const x = this.xs[this.currentDataIndex];
    const y = this.yt[this.currentDataIndex];
    this.currentOutput = this._network.forward(x.map(val => new Value(val)));

    this.currentBatchInputs.push(x);
    this.currentBatchTargets.push(y);

    const result: Prediction = {
        input: x,
        output: this.currentOutput.map(v => v.data),
    };

    this.currentDataIndex++;
    return result;
  }

  calculateLoss(): Value | number | null {
    if (this.currentBatchInputs.length === 0) {
      console.error("No data in the current batch");
      return null;
    }

    this.currentLoss = this.calculateBatchLoss(this.currentBatchInputs, this.currentBatchTargets);

    console.log('Calculated loss:', this.currentLoss.data);


    return this.currentLoss;
  }


  private calculateBatchLoss(inputs: number[][], targets: number[]): Value {
    let totalLoss = new Value(0);
    for (let i = 0; i < inputs.length; i++) {
      const pred = this._network.forward(inputs[i].map(val => new Value(val)))[0];
      const target = new Value(targets[i]);
      const diff = pred.sub(target);
      const squaredDiff = diff.mul(diff);
      console.log(`Prediction: ${pred.data}, Target: ${target.data}, Squared Diff: ${squaredDiff.data}`);
      totalLoss = totalLoss.add(squaredDiff);
    }
    const avgLoss = totalLoss.div(new Value(inputs.length));
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

  stepBackwardAndGetGradientsGroupedByConnection(): BackwardStepGradientsPerConnecrion[] {


    if (!this.currentLoss) {
      // Recalculate the loss before each backward step
      this.calculateLoss();
      return [];
    }

    // Zero out existing gradients
    this._network.zeroGrad();

    // Perform backpropagation
    this.currentLoss.backward();

    const gradients: BackwardStepGradientsPerConnecrion[] = [];

    // Iterate through each layer and neuron to collect gradients
    this._network.layers.forEach((layer, layerIndex) => {
      layer.neurons.forEach((neuron, neuronIndex) => {
        neuron.w.forEach((weight, weightIndex) => {
          const fromNodeId = layerIndex === 0 ? `input_${weightIndex}` : `layer${layerIndex - 1}_neuron${weightIndex}`;
          const toNodeId = `layer${layerIndex}_neuron${neuronIndex}`;
          const connectionId = `conn_${fromNodeId}_to_${toNodeId}`;

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

}