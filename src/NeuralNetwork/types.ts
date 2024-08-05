export type ActivationFunction = 'tanh' | 'relu' | 'sigmoid' | 'linear';

export interface NeuronData {
  id: string;
  weights: number[];
  bias: number;
  activation: ActivationFunction;
}

export interface LayerData {
  id: string;
  neurons: NeuronData[];
}

export interface NetworkData {
  layers: LayerData[];
}