import { MLPConfig } from "./NeuralNetwork/types";

export const CONFIG = {
  INITIAL_NETWORK: {
    inputSize: 3,
    layers: [4, 4, 1],
    activations: ['identity', 'tanh', 'tanh']
  } as MLPConfig,
  INITIAL_TRAINING: {
    learningRate: 0.01,
    epochs: 1000,
    batchSize: 1
  },
  VISUALIZATION: {
    width: 800,
    height: 600,
    nodeWidth: 60,
    nodeHeight: 40,
    layerSpacing: 200,
    nodeSpacing: 80
  }
};