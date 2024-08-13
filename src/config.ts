import { MLPConfig } from "./NeuralNetwork/types";

export const CONFIG = {
  INITIAL_NETWORK: {
    inputSize: 2,
    layers: [5, 1],
    activations: ['tanh', 'identity']
  } as MLPConfig,
  INITIAL_TRAINING: {
    learningRate: 0.01,
    epochs: 1,
    batchSize: 4
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