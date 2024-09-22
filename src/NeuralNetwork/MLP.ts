import { ActivationFunction, NetworkData, MLPConfig } from './types';
import { Layer } from './layer';
import { Value } from './value';

export class MLP {
  layers: Layer[];
  activations: ActivationFunction[];
  inputSize: number;
  layerOutputs: Value[][] = [];
  private gradientMapping: { neuron: number; parameter: number }[] = [];
  private _initialInput: Value[] = [];

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
    this.computeGradientMapping();
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
    console.log("MLP forward pass starting...");
    this.clearLayerOutputs();
    this._initialInput = x.map(Value.from);
    let out: Value[] = this._initialInput;
    console.log("Input:", out.map(v => v.data));
    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i];
      console.log(`Processing layer ${i + 1}...`);
      out = layer.forward(out);
      console.log(`Layer ${i + 1} outputs:`, out.map(v => v.data));
      this.layerOutputs.push(out);
    }
  

    console.log("MLP forward pass completed.");
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
    
    // Deep copy weights and biases
    this.layers.forEach((layer, i) => {
      layer.neurons.forEach((neuron, j) => {
        neuron.w.forEach((w, k) => {
          newMLP.layers[i].neurons[j].w[k] = new Value(w.data, [], 'weight');
        });
        newMLP.layers[i].neurons[j].b = new Value(neuron.b.data, [], 'bias');
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

  private computeGradientMapping() {
    const parametersPerNeuron = this.getParametersPerNeuron();
    let index = 0;
  
    for (const { neuron, weights, bias } of parametersPerNeuron) {
      for (let parameter = 0; parameter < weights + bias; parameter++) {
        this.gradientMapping.push({ neuron, parameter });
        index++;
      }
    }
  }

  getGradientMapping() {
    return this.gradientMapping;
  }

  getParametersPerNeuron(): { neuron: number, weights: number, bias: number }[] {
    return this.layers.flatMap(layer => layer.getParametersCount());
  }
}