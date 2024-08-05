import ModuleBase from "./ModuleBase";
import { Value } from "./Value";

export default class Neuron extends ModuleBase {
  w: Value[];
  b: Value;
  activation: string;

  constructor(nin: number, activation: string = 'ReLU') {
    super();
    this.w = Array.from({ length: nin }, () => new Value(Math.random() * 2 - 1));
    this.b = new Value(0);
    this.activation = activation;
  }

  activate(x: Value): Value {
    switch (this.activation) {
      case 'ReLU':
        return x.relu();
      case 'Sigmoid':
        return x.sigmoid();
      case 'Tanh':
        return x.tanh();
      case 'LeakyReLU':
        return x.leakyRelu();
      default:
        return x;
    }
  }

  forward(x: Value[]): Value {
    const act = this.w.reduce((acc, wi, i) => acc.add(wi.mul(x[i])), this.b);
    return this.activate(act);
  }

  parameters(): Value[] {
    return [...this.w, this.b];
  }

  toString(): string {
    return `${this.activation}Neuron(${this.w.length})`;
  }
}