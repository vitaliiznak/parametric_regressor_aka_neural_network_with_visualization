import { MLPConfig } from "./NeuralNetwork/types";

export const CONFIG = {
  INITIAL_NETWORK: {
    inputSize: 1,
    layers: [8, 8, 1],
    activations: ['leaky-relu', 'leaky-relu',  'identity']
  } as MLPConfig,
  INITIAL_TRAINING: {
    learningRate: 0.001,
    defaultBatchSize: 32,
    defaultEpochs: 70,
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