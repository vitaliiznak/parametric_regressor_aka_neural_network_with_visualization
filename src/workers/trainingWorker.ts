import { Trainer } from '../trainer';

self.onmessage = async (e) => {
  const { network, config, xs, yt } = e.data;
  const trainer = new Trainer(network, config);

  for await (const result of trainer.train(xs, yt)) {
    self.postMessage(result);
  }
};