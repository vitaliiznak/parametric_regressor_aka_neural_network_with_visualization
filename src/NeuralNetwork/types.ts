export type ActivationFunction = 'tanh' | 'relu' | 'sigmoid' | 'identity' | 'leaky-relu';

export interface NeuronData {
  id: string;
  weights: number[];
  bias: number;
  activation: ActivationFunction;
  name: string;
}

export interface LayerData {
  id: string;
  neurons: NeuronData[];
}

export interface NetworkData {
  // @TODO: inputSize should be inferred from the first layer
  inputSize: number;
  layers: LayerData[];
}

export interface MLPConfig {
  inputSize: number;
  layers: number[];
  activations: ActivationFunction[];
}

export interface SerializableNetwork {
  inputSize: number;
  layers: number[];
  activations?: ActivationFunction[];
  weights?: number[][][];
  biases?: number[][];
}