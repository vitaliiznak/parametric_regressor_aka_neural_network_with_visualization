import { ActivationFunction, NetworkData, MLPConfig } from './types';
import { Layer } from './Layer';
import { Value } from './Value';

export class MLP {
  private layers: Layer[];
  private activations: ActivationFunction[];
  private inputSize: number;
  private layerOutputs: Value[][] = [];
  private gradientMapping: Array<{ neuron: number; parameter: number }> = [];
  private _initialInput: Value[] = [];

  /**
   * Creates a new Multi-Layer Perceptron
   * @param config Network configuration
   */
  constructor(config: MLPConfig) {
    this.validateConfig(config);
    
    const { inputSize, layers, activations } = config;
    this.inputSize = inputSize;
    const sizes = [inputSize, ...layers];
    this.activations = activations || [];
    this.layers = [];

    for (let i = 0; i < layers.length; i++) {
      this.layers.push(
        new Layer(sizes[i], layers[i], this.activations[i] || 'identity')
      );
    }

    this.initializeGradientMapping();
  }

  /**
   * Validates the network configuration
   * @param config Network configuration to validate
   * @throws Error if configuration is invalid
   */
  private validateConfig(config: MLPConfig): void {
    if (!config.inputSize || config.inputSize <= 0) {
      throw new Error('Invalid input size');
    }
    if (!config.layers || !config.layers.length) {
      throw new Error('At least one layer is required');
    }
    if (config.activations && config.activations.length !== config.layers.length) {
      throw new Error('Number of activation functions must match number of layers');
    }
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
    this.clearLayerOutputs();
    this._initialInput = x.map(Value.from);
    let out: Value[] = this._initialInput;
    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i];
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

  private initializeGradientMapping() {
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