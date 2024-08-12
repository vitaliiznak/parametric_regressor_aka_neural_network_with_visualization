import { NetworkData } from '../NeuralNetwork/types';
import { SimulationOutput } from '../store';
import { VisualNetworkData, VisualNode, VisualConnection } from './types';

export class NetworkLayout {
  public nodeWidth = 60;
  public nodeHeight = 40;
  public layerSpacing = 210;
  public nodeSpacing = 80;
  public inputValuesSpacing = 10;
  public inputValueAndNetworkSpacing = 180;

  constructor(public canvasWidth: number, public canvasHeight: number) { }

  calculateLayout(network: NetworkData, simulationOutput?: SimulationOutput): VisualNetworkData {
    const nodes: VisualNode[] = [];
    const connections: VisualConnection[] = [];
    console.log('Network data:', network);
    console.log('network.inputSize', network.inputSize);
    const inputSize = network.inputSize;
    console.log('Calculated inputSize:', inputSize);

    const layerCount = network.layers.length;
    const maxNeuronsInLayer = Math.max(inputSize, ...network.layers.map(layer => layer.neurons.length));

    const totalWidth = (layerCount + 1) * this.layerSpacing;
    const totalHeight = (maxNeuronsInLayer + 1) * (this.nodeHeight + this.nodeSpacing);

    this.canvasWidth = Math.max(this.canvasWidth, totalWidth);
    this.canvasHeight = Math.max(this.canvasHeight, totalHeight);

    // Calculate startY for input nodes
    const startY = (this.canvasHeight - totalHeight) / 2;
    for (let i = 0; i < inputSize; i++) {
      for (let i = 0; i < inputSize; i++) {
        nodes.push({
          id: `input_${i}`,
          label: ['Size', 'Bedrooms', 'Age'][i],
          layerId: 'input',
          x: this.inputValuesSpacing,
          y: startY + i * (this.nodeHeight + this.nodeSpacing),
          weights: [0],
          bias: 0
        });
      }
    }

    network.layers.forEach((layer, layerIndex) => {
      const x = this.inputValueAndNetworkSpacing + layerIndex * this.layerSpacing + this.layerSpacing / 2;
      const layerHeight = layer.neurons.length * this.nodeHeight + (layer.neurons.length - 1) * this.nodeSpacing;
      const startY = (this.canvasHeight - layerHeight) / 2;

      layer.neurons.forEach((neuron, neuronIndex) => {
        const node: VisualNode = {
          id: neuron.id,
          label: `N${neuronIndex}`,
          layerId: layer.id,
          x: x - this.nodeWidth / 2,
          y: startY + neuronIndex * (this.nodeHeight + this.nodeSpacing),
          activation: neuron.activation,
          weights: neuron.weights,
          bias: neuron.bias
        };
        nodes.push(node);

        if (layerIndex === 0) {
          // Connect input nodes to first layer
          for (let i = 0; i < inputSize; i++) {
            connections.push({
              from: `input_${i}`,
              to: neuron.id,
              weight: neuron.weights[i],
              bias: neuron.bias
            });
          }
        } else {
          network.layers[layerIndex - 1].neurons.forEach((prevNeuron, prevIndex) => {
            connections.push({
              from: prevNeuron.id,
              to: neuron.id,
              weight: neuron.weights[prevIndex],
              bias: neuron.bias
            });
          });
        }
      });
    });

    console.log('Generated nodes:', nodes);
    console.log('Generated connections:', connections);

    
    if (simulationOutput) {
      const { input, output, layerOutputs } = simulationOutput;
    
      nodes.forEach((node) => {
        const [nodeType, layerIndexStr, nodeIndexStr] = node.id.split('_');
        const layerIndex = parseInt(layerIndexStr);
        const nodeIndex = parseInt(nodeIndexStr);
    
        if (nodeType === 'input' && input[nodeIndex] !== undefined) {
          node.outputValue = input[nodeIndex];
        } else if (nodeType === 'neuron') {
          if (layerOutputs[layerIndex] && layerOutputs[layerIndex][nodeIndex] !== undefined) {
            node.outputValue = layerOutputs[layerIndex][nodeIndex];
          }
          // Set output values for the last layer
          if (layerIndex === layerOutputs.length - 1 && output[nodeIndex] !== undefined) {
            node.outputValue = output[nodeIndex];
          }
        }
      });
    }

    return { nodes, connections };
  }

  findNodeAt(
    x: number,
    y: number,
    nodes: VisualNode[],
    scale: number,
    offsetX: number,
    offsetY: number
  ): VisualNode | null {
    const scaledX = (x - offsetX) / scale;
    const scaledY = (y - offsetY) / scale;

    for (const node of nodes) {
      if (
        scaledX >= node.x &&
        scaledX <= node.x + this.nodeWidth &&
        scaledY >= node.y &&
        scaledY <= node.y + this.nodeHeight
      ) {
        return node;
      }
    }
    return null;
  }
}