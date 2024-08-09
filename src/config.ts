import { MLPConfig } from "./NeuralNetwork/types";

export const CONFIG = {
  INITIAL_NETWORK: {
    inputSize: 1,
    layers: [1, 1],
    activations: ['identity']
  } as MLPConfig,
  INITIAL_TRAINING: {
    learningRate: 0.01,
    epochs: 1,
    batchSize: 4
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