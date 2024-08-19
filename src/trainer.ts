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
        yield { step: 'iteration', data: { iteration: this.currentIteration, loss: result.data.loss } };
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

    const backwardResult: TrainingResult = {
      step: 'backward',
      data: { gradients: this._network.parameters().map(p => p.grad) }
    };
    this.currentStep++;

    const oldWeights = this._network.parameters().map(p => p.data);
    this._network.parameters().forEach(p => {
      p.data -= this.config.learningRate * p.grad;
    });

    console.log('Weights updated');
    console.log('Old weights:', oldWeights);
    console.log('New weights:', this._network.parameters().map(p => p.data));

    const updateResult: TrainingResult = {
      step: 'update',
      data: {
        oldWeights,
        newWeights: this._network.parameters().map(p => p.data),
        learningRate: this.config.learningRate
      }
    };
    this.currentStep++;

    const iterationResult: TrainingResult = {
      step: 'iteration',
      data: { iteration: this.currentIteration, loss: loss.data }
    };

    console.log(`--- End of Training Step ${this.currentStep} ---\n`);

    return iterationResult;
  }


  singleStepForward(): TrainingResult | null {
    if (this.currentIteration >= this.config.iterations) {
      return null;
    }

    const x = this.xs[this.currentBatch];
    this.currentInput = x;
    this.currentOutput = this._network.forward(x.map(val => new Value(val)));

    const result: TrainingResult = {
      step: 'forward',
      data: {
        input: x,
        output: this.currentOutput.map(v => v.data),
        iteration: this.currentIteration,
        batchIndex: this.currentBatch,
        stepIndex: this.currentStep
      }
    };

    this.history.push(result);
    this.currentStep++;
    return result;
  }

  stepBackward(): TrainingResult | null {
    if (!this.currentOutput || !this.currentInput) {
      return null;
    }

    const target = new Value(this.yt[this.currentBatch]);
    this.currentLoss = this.currentOutput[0].sub(target).pow(2);
    this._network.zeroGrad();
    this.currentLoss.backward();

    const result: TrainingResult = {
      step: 'backward',
      data: {
        loss: this.currentLoss.data,
        gradients: this._network.parameters().map(p => p.grad),
      }
    };

    this.history.push(result);
    return result;
  }

  updateWeights(): TrainingResult | null {
    if (!this.currentLoss) {
      return null;
    }

    const oldWeights = this._network.parameters().map(p => p.data);
    this._network.parameters().forEach(p => {
      p.data -= this.config.learningRate * p.grad;
    });

    const result: TrainingResult = {
      step: 'update',
      data: {
        oldWeights,
        newWeights: this._network.parameters().map(p => p.data),
        learningRate: this.config.learningRate,
      }
    };

    this.history.push(result);
    this.currentBatch++;
    this.currentStep++;

    return result;
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
      step: 'iteration',
      data: {
        iteration: this.currentIteration,
        loss: this.currentLoss?.data
      }
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