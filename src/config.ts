import { MLPConfig } from "./NeuralNetwork/types";

export const CONFIG = {
  INITIAL_NETWORK: {
    inputSize: 1,
    layers: [5, 3, 1],
    activations: ['tanh', 'leaky-relu', 'identity']
  } as MLPConfig,
  INITIAL_TRAINING: {
    learningRate: 0.005
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