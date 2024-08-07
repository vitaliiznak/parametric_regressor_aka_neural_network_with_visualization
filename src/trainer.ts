import { MLP } from "./NeuralNetwork/mlp";
import { Value } from "./NeuralNetwork/value";


export interface TrainingConfig {
  learningRate: number;
  epochs: number;
  batchSize: number;
}

export type TrainingResult = {
  step: 'forward' | 'loss' | 'backward' | 'update' | 'epoch';
  data: {
    input?: number[];
    output?: Value | Value[];
    loss?: number;
    gradients?: number[];
    oldWeights?: number[];
    updatedWeights?: number[];
    epoch?: number;
    newWeights?: number[];
    learningRate?: number;
  };
};

export class Trainer {
  constructor(private network: MLP, private config: TrainingConfig) {}

  async* train(xs: number[][], yt: number[]): AsyncGenerator<TrainingResult> {
    for (let epoch = 0; epoch < this.config.epochs; epoch++) {
      let totalLoss = new Value(0);

      for (let i = 0; i < xs.length; i += this.config.batchSize) {
        const batchXs = xs.slice(i, i + this.config.batchSize);
        const batchYt = yt.slice(i, i + this.config.batchSize);

        // Forward pass
        const ypred = [];
        for (const x of batchXs) {
          const output = this.network.forward(x.map(val => new Value(val)));
          ypred.push(Array.isArray(output) ? output[0] : output);
          yield { step: 'forward', data: { input: x, output: output  } };
        }

        // Loss calculation
        const loss = ypred.reduce((sum, ypred_el, j) => {
          const diff = ypred_el.add(new Value(-batchYt[j]));
          return sum.add(diff.mul(diff));
        }, new Value(0));
        yield { step: 'loss', data: { loss: loss.data } };

        totalLoss = totalLoss.add(loss);

        // Backward pass
        this.network.zeroGrad();
        loss.backward();
        yield { step: 'backward', data: { gradients: this.network.parameters().map(p => p.grad) } };

        // Weight update
        const oldWeights = this.network.parameters().map(p => p.data);
        this.network.parameters().forEach(p => {
          p.data -= this.config.learningRate * p.grad;
        });
        yield { 
          step: 'update', 
          data: { 
            oldWeights,
            newWeights: this.network.parameters().map(p => p.data),
            learningRate: this.config.learningRate
          } 
        };
      }

      yield {
        step: 'epoch',
        data: {
          epoch: epoch + 1,
          loss: totalLoss.data / xs.length
        }
      };
    }
  }
  
}