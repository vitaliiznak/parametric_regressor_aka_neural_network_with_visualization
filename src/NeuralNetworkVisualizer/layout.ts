import { NetworkData } from '../NeuralNetwork/types';
import { VisualNetworkData, VisualNode, VisualConnection } from './types';

export class NetworkLayout {
  public nodeWidth = 60;
  public nodeHeight = 40;
  public layerSpacing = 200;
  public nodeSpacing = 80;

  constructor(public canvasWidth: number, public canvasHeight: number) {}

  calculateLayout(network: NetworkData): VisualNetworkData {
    const nodes: VisualNode[] = [];
    const connections: VisualConnection[] = [];

    const layerCount = network.layers.length;
    const maxNeuronsInLayer = Math.max(...network.layers.map(layer => layer.neurons.length));
    
    const totalWidth = (layerCount + 1) * this.layerSpacing;
    const totalHeight = (maxNeuronsInLayer + 1) * (this.nodeHeight + this.nodeSpacing);

    this.canvasWidth = Math.max(this.canvasWidth, totalWidth);
    this.canvasHeight = Math.max(this.canvasHeight, totalHeight);
  
    network.layers.forEach((layer, layerIndex) => {
      const x = layerIndex * this.layerSpacing + this.layerSpacing / 2;
      const layerHeight = layer.neurons.length * this.nodeHeight + (layer.neurons.length - 1) * this.nodeSpacing;
      const startY = (this.canvasHeight - layerHeight) / 2;

      layer.neurons.forEach((neuron, neuronIndex) => {
        const node: VisualNode = {
          id: neuron.id,
          label: `N${neuronIndex}`,
          layerId: layer.id,
          x: x - this.nodeWidth / 2,
          y: startY + neuronIndex * (this.nodeHeight + this.nodeSpacing)
        };
        nodes.push(node);

        if (layerIndex > 0) {
          network.layers[layerIndex - 1].neurons.forEach((prevNeuron, prevIndex) => {
            connections.push({
              from: prevNeuron.id,
              to: neuron.id,
              weight: neuron.weights[prevIndex]
            });
          });
        }
      });
    });

    return { nodes, connections };
  }

  findNodeAt(x: number, y: number, nodes: VisualNode[]): VisualNode | null {
    return nodes.find(node =>
      x >= node.x && x <= node.x + this.nodeWidth &&
      y >= node.y && y <= node.y + this.nodeHeight
    ) || null;
  }
}