import { ActivationFunction, NetworkData, LayerData } from './types';
import { Layer } from './layer';
import { Value } from './value';

export class MLP {
  layers: Layer[];
  activations: ActivationFunction[];

  constructor(nin: number, nouts: number[], activations: ActivationFunction[] = []) {
    const sizes = [nin, ...nouts];
    this.activations = activations;
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

  toJSON(): NetworkData {
    return {
      layers: this.layers.map((layer, layerIndex) => ({
        id: `layer_${layerIndex}`,
        activations: this.activations,
        neurons: layer.neurons.map((neuron, neuronIndex) => ({
          id: `neuron_${layerIndex}_${neuronIndex}`,
          weights: neuron.w.map(w => w.data),
          bias: neuron.b.data,
          activation: neuron.activation
        }))
      }))
    };
  }
}