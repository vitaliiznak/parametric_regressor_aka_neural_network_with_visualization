import { MLP } from "./NeuralNetwork/mlp";
import { Value } from "./NeuralNetwork/value";

export interface TrainingConfig {
  learningRate: number;
  epochs: number;
  batchSize: number;
}

export interface TrainingResult {
  step: 'forward' | 'loss' | 'backward' | 'update' | 'epoch';
  data: any;
}

export class Trainer {
  private network: MLP;
  private config: TrainingConfig;
  private currentEpoch: number = 0;
  private currentBatch: number = 0;
  private currentStep: number = 0;
  private xs: number[][] = [];
  private yt: number[] = [];
  private history: TrainingResult[] = [];

  constructor(network: MLP, config: TrainingConfig) {
    this.network = network.clone(); 
    this.config = config;
  }

  getNetwork(): MLP {
    return this.network;
  }

  async* train(xs: number[][], yt: number[]): AsyncGenerator<TrainingResult, void, unknown> {
    this.xs = xs;
    this.yt = yt;
    this.currentEpoch = 0;
    this.currentBatch = 0;
    this.currentStep = 0;
    this.history = [];

    while (this.currentEpoch < this.config.epochs) {
      const batchXs = this.xs.slice(this.currentBatch, this.currentBatch + this.config.batchSize);
      const batchYt = this.yt.slice(this.currentBatch, this.currentBatch + this.config.batchSize);

      const result = await this.trainStep(batchXs, batchYt);
      this.history.push(result);
      yield result;

      this.currentBatch += this.config.batchSize;
      if (this.currentBatch >= this.xs.length) {
        this.currentBatch = 0;
        this.currentEpoch++;
      }
    }
  }

  private async trainStep(batchXs: number[][], batchYt: number[]): Promise<TrainingResult> {
    console.log('batchXs', batchXs);
    console.log(`\n--- Training Step ${this.currentStep} ---`);
    console.log(`Epoch: ${this.currentEpoch + 1}/${this.config.epochs}, Batch: ${this.currentBatch / this.config.batchSize + 1}`);

    const ypred = batchXs.map(x => {
      const result = this.network.forward(x.map(val => new Value(val)));
      return Array.isArray(result) ? result[0] : result;
    });

    console.log('Forward pass completed');
    console.log('Predictions:', ypred.map(y => y.data));
    console.log('Targets:', batchYt);

    this.currentStep++;
    
    const loss = ypred.reduce((sum, ypred_el, j) => {
      const target = new Value(batchYt[j]);
      const diff = ypred_el.sub(target);
      return sum.add(diff.mul(diff));
    }, new Value(0));

    console.log('Loss calculated:', loss.data);

    this.currentStep++;

    this.network.zeroGrad();
    loss.backward();

    console.log('Backward pass completed');
    console.log('Gradients:', this.network.parameters().map(p => p.grad));

    const backwardResult: TrainingResult = {
      step: 'backward',
      data: { gradients: this.network.parameters().map(p => p.grad) }
    };
    this.currentStep++;

    const oldWeights = this.network.parameters().map(p => p.data);
    this.network.parameters().forEach(p => {
      p.data -= this.config.learningRate * p.grad;
    });

    console.log('Weights updated');
    console.log('Old weights:', oldWeights);
    console.log('New weights:', this.network.parameters().map(p => p.data));

    const updateResult: TrainingResult = {
      step: 'update',
      data: {
        oldWeights,
        newWeights: this.network.parameters().map(p => p.data),
        learningRate: this.config.learningRate
      }
    };
    this.currentStep++;

    const epochResult: TrainingResult = {
      step: 'epoch',
      data: { epoch: this.currentEpoch, loss: loss.data }
    };

    console.log(`--- End of Training Step ${this.currentStep} ---\n`);

    return epochResult;
  }

  stepForward(): TrainingResult | null {
    if (this.currentStep >= this.history.length) {
      return null;
    }
    return this.history[this.currentStep++];
  }

  stepBackward(): TrainingResult | null {
    if (this.currentStep <= 0) {
      return null;
    }
    return this.history[--this.currentStep];
  }
}