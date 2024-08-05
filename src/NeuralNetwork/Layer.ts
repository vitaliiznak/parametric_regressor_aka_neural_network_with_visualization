import ModuleBase from "./ModuleBase";
import Neuron from "./Neuron";
import Value from "./Value";

export default class Layer extends ModuleBase {
  neurons: Neuron[];

  constructor(nin: number, nout: number, activation: string = 'ReLU') {
    super();
    this.neurons = Array.from({ length: nout }, () => new Neuron(nin, activation));
  }

  forward(x: Value[]): Value[] {
    return this.neurons.map(n => n.forward(x));
  }

  parameters(): Value[] {
    return this.neurons.flatMap(n => n.parameters());
  }

  toString(): string {
    return `Layer of [${this.neurons.join(', ')}]`;
  }
}
