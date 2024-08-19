import { Trainer } from '../trainer';
import { MLP } from '../NeuralNetwork/mlp';
import { TrainingConfig } from '../types';
import { Value } from '../NeuralNetwork/value';
import { NetworkData, SerializableNetwork } from '../NeuralNetwork/types';

let network: SerializableNetwork;
let config: TrainingConfig;
let xs: number[][] = [];
let ys: number[] = [];
let trainer: Trainer | null = null;

self.onmessage = async (e: MessageEvent) => {
  const { type, data } = e.data;

  console.log(`Received ${type} data in worker:`, data);

  switch (type) {
    case 'network':
      network = data as SerializableNetwork;
      break;
    case 'config':
      config = data;
      break;
    case 'trainingData':
      const parsedData = JSON.parse(data);
      xs = parsedData.xs;
      ys = parsedData.ys;
      console.log('Received training data:', { xsLength: xs.length, ysLength: ys.length });
      startTraining();
      break;
    case 'pause':
      if (trainer) trainer.pause();
      break;
    case 'resume':
      if (trainer) trainer.resume();
      break;
    case 'stop':
      if (trainer) trainer.stop();
      break;
  }
};

async function startTraining() {
  console.log('Starting training with network:', network);
  console.log('Config:', config);
  const mlp = new MLP({
    inputSize: network.inputSize,
    layers: network.layers,
    activations: network.activations || []
  });
  
  // Set weights and biases
  mlp.layers.forEach((layer, i) => {
    const inputSize = i === 0 ? network.inputSize : network.layers[i - 1];
    layer.neurons.forEach((neuron, j) => {
      if (network.weights && network.weights[i] && network.weights[i][j]) {
        neuron.w = network.weights[i][j].map(w => new Value(w));
      } else {
        // Initialize with random weights if not provided
        neuron.w = Array(inputSize).fill(0).map(() => new Value(Math.random() - 0.5));
      }
      
      if (network.biases && network.biases[i] && network.biases[i][j] !== undefined) {
        neuron.b = new Value(network.biases[i][j]);
      } else {
        // Initialize with a random bias if not provided
        neuron.b = new Value(Math.random() - 0.5);
      }
    });
  });

  trainer = new Trainer(mlp, config);

  for await (const result of trainer.train(xs, ys)) {
    self.postMessage({ 
      type: 'progress', 
      data: {
        ...result,
        network: mlp.toJSON()
      }
    });

    // Add a small delay to prevent UI freezing
    await new Promise(resolve => setTimeout(resolve, 0));
  }

  self.postMessage({ type: 'complete', data: mlp.toJSON() });
  trainer = null;
}