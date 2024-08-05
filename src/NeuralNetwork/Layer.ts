import { Neuron, ActivationFunction } from './Neuron';
import { Value } from './Value';

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

  toString(): string {
    return `Layer of [${this.neurons.join(', ')}]`;
  }
}