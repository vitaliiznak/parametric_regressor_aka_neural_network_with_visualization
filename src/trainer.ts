import { MLP } from "./NeuralNetwork/mlp";
import { Value } from "./NeuralNetwork/value";
import { TrainingConfig, TrainingResult } from "./types";

export class Trainer {
  _network: MLP;
  private config: TrainingConfig;
  private currentIteration: number = 0;
  private currentBatch: number = 0;
  private currentStep: number = 0;
  private xs: number[][] = [];
  private yt: number[] = [];
  private history: TrainingResult[] = [];
  private currentInput: number[] | null = null;
  private currentOutput: Value[] | null = null;
  private currentLoss: Value | null = null;
  private isPaused: boolean = false;
  private currentDataIndex: number = 0;

  constructor(network: MLP, config: TrainingConfig) {
    this._network = network.clone();
    this.config = config;
  }

  get network(): MLP {
    return this._network;
  }

  setTrainingData(xs: number[][], yt: number[]): void {
    this.xs = xs;
    this.yt = yt;
    this.currentIteration = 0;
    this.currentBatch = 0;
    this.currentStep = 0;
    this.history = [];
  }


  async *train(xs: number[][], yt: number[]): AsyncGenerator<TrainingResult, void, unknown> {
    this.xs = xs;
    this.yt = yt;
    this.currentIteration = 0;
    this.currentBatch = 0;
    this.currentStep = 0;
    this.history = [];

    while (this.currentIteration < this.config.iterations) {
      while (this.isPaused) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      const batchXs = this.xs.slice(this.currentBatch, this.currentBatch + this.config.batchSize);
      const batchYt = this.yt.slice(this.currentBatch, this.currentBatch + this.config.batchSize);

      const result = await this.trainStep(batchXs, batchYt);
      this.history.push(result);
      yield result;

      this.currentBatch += this.config.batchSize;
      this.currentStep++;

      if (this.currentBatch >= this.xs.length) {
        this.currentBatch = 0;
        this.currentIteration++;
        yield {};
      }
    }
  }

  private async trainStep(batchXs: number[][], batchYt: number[]): Promise<TrainingResult> {
    console.log('batchXs', batchXs);
    console.log(`\n--- Training Step ${this.currentStep} ---`);
    console.log(`Iteration: ${this.currentIteration + 1}/${this.config.iterations}, Batch: ${this.currentBatch / this.config.batchSize + 1}`);

    // Update currentInput with the first input of the batch
    this.currentInput = batchXs[0];

    const ypred = batchXs.map(x => {
      const result = this._network.forward(x.map(val => new Value(val)));
      return result[0]; // Update here
    });

    console.log('Forward pass completed');
    console.log('Predictions:', ypred.map(y => y.data));
    console.log('Targets:', batchYt);

    this.currentStep++;
    
    const loss = ypred.reduce((sum, ypred_el, j) => {
      const target = new Value(batchYt[j]);
      const diff = ypred_el.sub(target);
      return sum.add(diff.mul(diff));
    }, new Value(0)).div(new Value(batchYt.length)); // MSE

    console.log('Loss calculated:', loss.data);

    this.currentStep++;

    this._network.zeroGrad();
    loss.backward();

    console.log('Backward pass completed');
    console.log('Gradients:', this._network.parameters().map(p => p.grad));

    this.currentStep++;

    const oldWeights = this._network.parameters().map(p => p.data);
    this._network.parameters().forEach(p => {
      p.data -= this.config.learningRate * p.grad;
    });

    console.log('Weights updated');
    console.log('Old weights:', oldWeights);
    console.log('New weights:', this._network.parameters().map(p => p.data));

    this.currentStep++;

    const iterationResult: TrainingResult = {
      data: { }
    };

    console.log(`--- End of Training Step ${this.currentStep} ---\n`);

    return iterationResult;
  }


  singleStepForward(): TrainingResult | null {
    if (this.currentDataIndex >= this.xs.length) {
      this.currentDataIndex = 0;
      return null;
    }

    const x = this.xs[this.currentDataIndex];
    this.currentInput = x;
    this.currentOutput = this._network.forward(x.map(val => new Value(val)));

    const result: TrainingResult = {
        input: x,
        output: this.currentOutput.map(v => v.data),
    };

    this.history.push(result);
    this.currentStep++;
    this.currentDataIndex++;
    return result;
  }

  calculateLoss(): TrainingResult | null {
    if (this.currentDataIndex < this.config.batchSize) {
      console.error("Not enough forward steps to calculate loss");
      return null;
    }

    const batchInputs = this.xs.slice(this.currentBatch, this.currentBatch + this.config.batchSize);
    const batchTargets = this.yt.slice(this.currentBatch, this.currentBatch + this.config.batchSize);

    const predictions = batchInputs.map(x => this._network.forward(x.map(val => new Value(val))));
    this.currentLoss = this.calculateBatchLoss(predictions, batchTargets);

    const result: TrainingResult = {
    };

    this.history.push(result);
    this.currentStep++;
    return result;
  }

  stepBackward(): TrainingResult | null {
    if (!this.currentLoss) {
      console.error("Loss not calculated");
      return null;
    }

    this._network.zeroGrad();
    this.currentLoss.backward();

    const result: TrainingResult = {
        gradients: this._network.parameters().map(p => p.grad),
    };

    this.history.push(result);
    return result;
  }

  updateWeights(): TrainingResult | null {
    if (!this.currentLoss) {
      console.error("Backward step not performed");
      return null;
    }

    const oldWeights = this._network.parameters().map(p => p.data);
    this._network.parameters().forEach(p => {
      p.data -= this.config.learningRate * p.grad;
    });

    const result: TrainingResult = {
        oldWeights,
        newWeights: this._network.parameters().map(p => p.data),
    };

    this.history.push(result);
    this.currentBatch += this.config.batchSize;
    this.currentStep++;

    if (this.currentBatch >= this.xs.length) {
      this.currentBatch = 0;
      this.currentIteration++;
    }

    return result;
  }

  private calculateBatchLoss(predictions: Value[][], targets: number[]): Value {
    let totalLoss = new Value(0);
    for (let i = 0; i < predictions.length; i++) {
      const pred = predictions[i][0];
      const target = new Value(targets[i]);
      const loss = pred.sub(target).pow(2);
      totalLoss = totalLoss.add(loss);
    }
    return totalLoss.div(new Value(predictions.length));
  }

  completeIteration(): TrainingResult | null {
    if (this.currentIteration >= this.config.iterations) {
      return null;
    }

    while (this.currentBatch < this.xs.length) {
      this.singleStepForward();
      this.stepBackward();
      this.updateWeights();
      this.currentBatch++;
    }

    this.currentBatch = 0;
    this.currentIteration++;

    const result: TrainingResult = {
  
    };

    this.history.push(result);
    return result;
  }

  getCurrentIteration(): number {
    return this.currentIteration;
  }

  getCurrentBatch(): number {
    return this.currentBatch;
  }

  getCurrentStep(): number {
    return this.currentStep;
  }

  isReadyForLossCalculation(): boolean {
    return this.currentBatch >= this.config.batchSize;
  }
}