import { ActivationFunction, NetworkData, MLPConfig } from './types';
import { Layer } from './layer';
import { Value } from './value';

export class MLP {
  layers: Layer[];
  activations: ActivationFunction[];
  inputSize: number;
  layerOutputs: Value[][] = [];

  constructor(config: MLPConfig) {
    const { inputSize, layers, activations } = config;
    this.inputSize = inputSize;
    const sizes = [inputSize, ...layers];
    this.activations = activations || [];
    this.layers = [];
    for (let i = 0; i < layers.length; i++) {
      this.layers.push(new Layer(sizes[i], layers[i], this.activations[i] || 'identity'));
    }
    console.log("Creating MLP with layers:", layers, "and activations:", this.activations);
    this.layers.forEach((layer, i) => {
      console.log(`Layer ${i}: size ${layer.neurons.length}, activation ${layer.neurons[0].activation}`);
    });
    this.clearLayerOutputs();
  }

  getLayerOutputs(): number[][] {
    return this.layerOutputs.map(layer => {
      return layer.map(v => v.data)
    });
  }

  clearLayerOutputs(): void {
    this.layerOutputs = [];
  }

  forward(x: (number | Value)[]): Value[] {
    this.clearLayerOutputs(); // Clear outputs before a new forward pass
    let out: Value[] = x.map(Value.from);
    for (const layer of this.layers) {
      out = layer.forward(out);
      this.layerOutputs.push(out);
    }
    return out;
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

  updateFromJSON(json: any) {
    // Update the network structure if necessary
    if (json.layers.length !== this.layers.length) {
      this.layers = json.layers.map((layer: any) => new Layer(layer.nin, layer.nout, layer.activation));
    }

    // Update weights and biases for each layer
    this.layers.forEach((layer, i) => {
      layer.neurons.forEach((neuron, j) => {
        neuron.w = json.layers[i].neurons[j].weights.map((w: number) => new Value(w));
        neuron.b = new Value(json.layers[i].neurons[j].bias);
      });
    });
  }
}