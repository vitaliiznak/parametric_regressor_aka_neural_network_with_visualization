import { MLP } from "./NeuralNetwork/mlp";
import { Value } from "./NeuralNetwork/value";


export interface TrainingConfig {
  learningRate: number;
  epochs: number;
  batchSize: number;
}

export interface TrainingResult {
  epoch: number;
  loss: number;
}

export class Trainer {
  constructor(private network: MLP, private config: TrainingConfig) {}

  async* train(xs: number[][], yt: number[]): AsyncGenerator<TrainingResult> {
    for (let epoch = 0; epoch < this.config.epochs; epoch++) {
      console.log(`Starting epoch ${epoch + 1}`);
      let totalLoss = new Value(0);
  
      for (let i = 0; i < xs.length; i += this.config.batchSize) {
        const batchXs = xs.slice(i, i + this.config.batchSize);
        const batchYt = yt.slice(i, i + this.config.batchSize);
  
        console.log(`Processing batch ${i / this.config.batchSize + 1}`);
        const ypred = batchXs.map(x => {
          const output = this.network.forward(x.map(val => new Value(val)));
          console.log(`Network output:`, output);
          return Array.isArray(output) ? output[0] : output;
        });
  
        const loss = ypred.reduce((sum, ypred_el, j) => {
          if (!(ypred_el instanceof Value)) {
            console.error('Unexpected output type:', ypred_el);
            return sum;
          }
          const diff = ypred_el.add(new Value(-batchYt[j]));
          return sum.add(diff.mul(diff));
        }, new Value(0));
  
        console.log(`Batch loss:`, loss.data);
        totalLoss = totalLoss.add(loss);
  
        this.network.zeroGrad();
        loss.backward();
  
        this.network.parameters().forEach(p => {
          p.data -= this.config.learningRate * p.grad;
        });
      }
  
      const epochResult = {
        epoch: epoch + 1,
        loss: totalLoss.data / xs.length
      };
      console.log(`Finished epoch ${epoch + 1}:`, epochResult);
      yield epochResult;
    }
  }
  
}