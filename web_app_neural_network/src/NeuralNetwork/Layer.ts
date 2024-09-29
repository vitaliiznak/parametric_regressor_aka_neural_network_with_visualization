
import { Neuron } from './neuron';
import { ActivationFunction } from './types';
import { Value } from './value';


export class Layer {
  neurons: Neuron[];

  constructor(nin: number, nout: number, activation: ActivationFunction = 'tanh') {
    this.neurons = Array(nout).fill(0).map(() => new Neuron(nin, activation));
  }

  forward(x: Value[]): Value[] {
    return this.neurons.map(neuron => neuron.forward(x));
  }

  parameters(): Value[] {
    return this.neurons.flatMap(neuron => neuron.parameters());
  }

 getParametersCount(): { neuron: number, weights: number, bias: number }[] {
    return this.neurons.map((neuron, index) => ({
      neuron: index + 1,
      weights: neuron.w.length,
      bias: 1
    }));
  }
}