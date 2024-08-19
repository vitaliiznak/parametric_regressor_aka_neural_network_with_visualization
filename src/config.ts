import { MLPConfig } from "./NeuralNetwork/types";

export const CONFIG = {
  INITIAL_NETWORK: {
    inputSize: 1,
    layers: [5, 1],
    activations: ['tanh', 'identity']
  } as MLPConfig,
  INITIAL_TRAINING: {
    learningRate: 0.01,
    iterations: 1,
    batchSize: 2
  },
  VISUALIZATION: {
    width: 1000,
    height: 800,
    nodeWidth: 60,
    nodeHeight: 40,
    layerSpacing: 200,
    nodeSpacing: 80
  }
};