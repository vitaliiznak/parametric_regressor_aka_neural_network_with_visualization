import { NetworkData } from '../NeuralNetwork/types';
import { SimulationResult, VisualConnection, VisualNetworkData, VisualNode } from '../types';


export class NetworkLayout {
  public nodeWidth = 50;
  public nodeHeight = 30;
  public inputNodeWidth = 100;
  public inputNodeHeight = 40;
  public layerSpacing = 180;
  public nodeSpacing = 60;
  public inputValuesSpacing = 10;
  public inputValueAndNetworkSpacing = 150;

  constructor(public canvasWidth: number, public canvasHeight: number) { }

  calculateLayout(
    network: NetworkData,
    currentInput: number[],
    simulationResult?: SimulationResult | null,
    customPositions?: Record<string, { x: number, y: number }>
  ): VisualNetworkData {
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
      nodes.push({
        id: `neuron_-1_${i}`, // Standardized ID
        label: `Input ${i}`,
        layerId: 'input',
        x: this.inputValuesSpacing,
        y: startY + i * (this.nodeHeight + this.nodeSpacing),
        weights: [0],
        bias: 0
      });
    }

    if (currentInput) {
      nodes.forEach((node, index) => {
        if (node.layerId === 'input' && currentInput[index] !== undefined) {
          node.outputValue = currentInput[index];
        }
      });
    }

    network.layers.forEach((layer, layerIndex) => {
      const x = this.inputValueAndNetworkSpacing + layerIndex * this.layerSpacing + this.layerSpacing / 2;
      const layerHeight = layer.neurons.length * this.nodeHeight + (layer.neurons.length - 1) * this.nodeSpacing;
      const startY = (this.canvasHeight - layerHeight) / 2;

      layer.neurons.forEach((neuron, neuronIndex) => {
        const nodeId = `neuron_${layerIndex}_${neuronIndex}`;
        const customPosition = customPositions?.[nodeId];

        const node: VisualNode = {
          id: nodeId,
          label: `N${neuronIndex}`,
          layerId: layer.id,
          x: customPosition ? customPosition.x : x - this.nodeWidth / 2,
          y: customPosition ? customPosition.y : startY + neuronIndex * (this.nodeHeight + this.nodeSpacing),
          activation: neuron.activation,
          weights: neuron.weights,
          bias: neuron.bias,
        };
        nodes.push(node);

        if (layerIndex === 0) {
          // Connect input nodes to first layer
          for (let i = 0; i < inputSize; i++) {
            connections.push({
              id: `from_neuron_-1_${i}_to_neuron_${layerIndex}_${neuronIndex}`, // Standardized format
              from: `neuron_-1_${i}`,
              to: nodeId,
              weight: neuron.weights[i],
              bias: neuron.bias
            });
          }
        } else {
          network.layers[layerIndex - 1].neurons.forEach((prevNeuron, prevIndex) => {
            connections.push({
              id: `from_neuron_${layerIndex - 1}_${prevIndex}_to_neuron_${layerIndex}_${neuronIndex}`, // Standardized format
              from: `neuron_${layerIndex - 1}_${prevIndex}`,
              to: nodeId,
              weight: neuron.weights[prevIndex],
              bias: neuron.bias
            });
          });
        }
      });
    });

    console.log('Generated nodes:', nodes);
    console.log('Generated connections:', connections);

    if (simulationResult) {
      const { input, layerOutputs } = simulationResult;
      nodes.forEach((node) => {
        const [nodeType, layerIndexStr, nodeIndexStr] = node.id.split('_');
        const layerIndex = parseInt(layerIndexStr);
        const nodeIndex = parseInt(nodeIndexStr);
    
        if (nodeType === 'neuron' && layerIndex === -1 && input[nodeIndex] !== undefined) {
          node.outputValue = input[nodeIndex];
        } else if (nodeType === 'neuron') {
          if (layerOutputs[layerIndex] && layerOutputs[layerIndex][nodeIndex] !== undefined) {
            node.outputValue = layerOutputs[layerIndex][nodeIndex];
            node.inputValues = layerIndex === 0 ? input : simulationResult?.layerOutputs?.[layerIndex - 1]
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

  updateDimensions(width: number, height: number) {
    this.canvasWidth = width;
    this.canvasHeight = height;
  }
}