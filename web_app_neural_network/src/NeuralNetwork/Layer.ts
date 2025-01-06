import { Neuron } from './Neuron';
import { ActivationFunction } from './types';
import { Value } from './Value';

export class Layer {
  neurons: Neuron[];

  /**
   * Creates a new neural network layer
   * @param nin Number of inputs to the layer
   * @param nout Number of neurons in the layer
   * @param activation Activation function to use
   */
  constructor(nin: number, nout: number, activation: ActivationFunction = 'tanh') {
    this.neurons = Array(nout)
      .fill(0)
      .map(() => new Neuron(nin, activation));
  }

  /**
   * Forward pass through the layer
   * @param x Input values
   * @returns Output values from each neuron
   * @throws Error if input dimensions don't match
   */
  forward(x: Value[]): Value[] {
    if (x.length !== this.neurons[0].w.length) {
      throw new Error(`Input dimension mismatch: expected ${this.neurons[0].w.length}, got ${x.length}`);
    }
    return this.neurons.map(neuron => neuron.forward(x));
  }

  /**
   * Get all trainable parameters in the layer
   */
  parameters(): Value[] {
    return this.neurons.flatMap(neuron => neuron.parameters());
  }

  /**
   * Get parameter count information for each neuron
   */
  getParametersCount(): Array<{
    neuron: number;
    weights: number;
    bias: number;
  }> {
    return this.neurons.map((neuron, index) => ({
      neuron: index + 1,
      weights: neuron.w.length,
      bias: 1
    }));
  }
}