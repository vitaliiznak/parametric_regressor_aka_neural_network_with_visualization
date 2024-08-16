import { Trainer } from '../trainer';
import { MLP } from '../NeuralNetwork/mlp';
import { TrainingConfig } from '../types';

self.onmessage = async (e: MessageEvent) => {
  const { network, config, xs, yt } = e.data;
  const mlp = new MLP(network);
  const trainer = new Trainer(mlp, config);

  for await (const result of trainer.train(xs, yt)) {
    self.postMessage({ 
      type: 'progress', 
      data: {
        ...result,
        network: mlp.toJSON()
      }
    });
  }

  self.postMessage({ type: 'complete', data: mlp.toJSON() });
};