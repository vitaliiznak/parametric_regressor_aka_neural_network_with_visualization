// neuron.ts
import { ActivationFunction, NeuronData } from './types';
import { Value } from './value';


export class Neuron {
  w: Value[];
  b: Value;
  activation: ActivationFunction;

  constructor(nin: number, activation: ActivationFunction = 'tanh') {
    this.w = Array(nin).fill(0).map(() => new Value(Math.random() * 2 - 1));
    this.b = new Value(0);
    this.activation = activation;
    console.log(`Neuron created with ${nin} inputs and ${this.activation} activation`);
  }

  forward(x: Value[]): Value {
    if (x.length !== this.w.length) {
      console.error(`Input size (${x.length}) doesn't match weight count (${this.w.length})`);
      // Adjust the input or weights to match
      while (x.length < this.w.length) {
        x.push(new Value(0));
      }
      while (this.w.length < x.length) {
        this.w.push(new Value(Math.random() * 2 - 1));
      }
    }

    const act = x.reduce((sum, xi, i) => sum.add(this.w[i].mul(xi)), this.b);
    return this.applyActivation(act);
  }

  applyActivation(x: Value): Value {
    switch (this.activation) {
      case 'tanh': return x.tanh();
      case 'relu': return x.relu();
      case 'sigmoid': return x.sigmoid();
      case 'identity': default: return x;
    }
  }

  parameters(): Value[] {
    return [...this.w, this.b];
  }

  toJSON(): NeuronData {
    return {
      id: Math.random().toString(36).substr(2, 9),
      weights: this.w.map(w => w.data),
      bias: this.b.data,
      activation: this.activation,
      name: this.activation
    };
  }
}