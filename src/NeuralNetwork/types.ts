export type ActivationFunction = 'tanh' | 'relu' | 'sigmoid' | 'identity';

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
  layers: LayerData[];
}

export interface MLPConfig {
  inputSize: number;
  layers: number[];
  activations: ActivationFunction[];
}