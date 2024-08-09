import { ActivationFunction, NetworkData, MLPConfig } from './types';
import { Layer } from './layer';
import { Value } from './value';

export class MLP {
  layers: Layer[];
  activations: ActivationFunction[];
  inputSize: number;

  constructor(config: MLPConfig) {
    const { inputSize, layers, activations } = config;
    this.inputSize = inputSize;
    const sizes = [inputSize, ...layers];
    this.activations = activations;
    this.layers = [];
    for (let i = 0; i < layers.length; i++) {
      this.layers.push(new Layer(sizes[i], layers[i], activations[i] || 'tanh'));
    }
    console.log("Creating MLP with layers:", layers, "and activations:", activations);
    this.layers.forEach((layer, i) => {
      console.log(`Layer ${i}: size ${layer.neurons.length}, activation ${layer.neurons[0].activation}`);
    });
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
      inputSize: this.inputSize,
      layers: this.layers.map((layer, layerIndex) => ({
        id: `layer_${layerIndex}`,
        neurons: layer.neurons.map((neuron, neuronIndex) => ({
          id: `neuron_${layerIndex}_${neuronIndex}`,
          weights: neuron.w.map(w => w.data),
          bias: neuron.b.data,
          activation: neuron.activation,
          name: neuron.activation
        }))
      }))
    };
  }

  clone(): MLP {
    const config: MLPConfig = {
      inputSize: this.inputSize,
      layers: this.layers.map(layer => layer.neurons.length),
      activations: this.activations
    };
    const newMLP = new MLP(config);
    
    // Copy weights and biases
    this.layers.forEach((layer, i) => {
      layer.neurons.forEach((neuron, j) => {
        neuron.w.forEach((w, k) => {
          newMLP.layers[i].neurons[j].w[k].data = w.data;
        });
        newMLP.layers[i].neurons[j].b.data = neuron.b.data;
      });
    });

    return newMLP;
  }
}