import { MLPConfig } from "./NeuralNetwork/types";

export const CONFIG = {
  INITIAL_NETWORK: {
    inputSize: 1,
    layers: [4, 2],
    activations: ['tanh','tanh', 'identity']
  } as MLPConfig,
  INITIAL_TRAINING: {
    learningRate: 0.02,
    defaultBatchSize: 20,
    defaultEpochs: 10,
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