import { Layer } from './Layer';
import { Value } from './Value';
import { ActivationFunction } from './Neuron';

export class MLP {
  layers: Layer[];

  constructor(nin: number, nouts: number[], activations: ActivationFunction[] = []) {
    const sizes = [nin, ...nouts];
    this.layers = sizes.slice(0, -1).map((s, i) => 
      new Layer(s, sizes[i+1], activations[i] || 'tanh')
    );
  }

  forward(x: (number | Value)[]): Value | Value[] {
    let out: Value[] = x.map(Value.from);
    for (const layer of this.layers) {
      out = layer.forward(out);
    }
    return out.length === 1 ? out[0] : out;
  }

  parameters(): Value[] {
    return this.layers.flatMap(layer => layer.parameters());
  }

  zeroGrad(): void {
    this.parameters().forEach(p => p.grad = 0);
  }

  toString(): string {
    return `MLP of [${this.layers.join(', ')}]`;
  }

  static fromConfig(config: {
    inputSize: number;
    hiddenSizes: number[];
    outputSize: number;
    activations?: ActivationFunction[];
  }): MLP {
    const { inputSize, hiddenSizes, outputSize, activations = [] } = config;
    return new MLP(inputSize, [...hiddenSizes, outputSize], activations);
  }
}