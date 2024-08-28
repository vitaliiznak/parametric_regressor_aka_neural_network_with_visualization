import { colors } from "../styles/colors";
import { Point, VisualConnection, VisualNetworkData, VisualNode } from "../types";
import { debounce } from "@solid-primitives/scheduled";

export class NetworkRenderer {
  private ctx: CanvasRenderingContext2D;
  public scale: number = 1;
  public offsetX: number = 0;
  public offsetY: number = 0;
  private debouncedRender: (data: VisualNetworkData, selectedNode: VisualNode | null) => void;
  public nodeWidth: number;
  public nodeHeight: number;
  private highlightedNodeId: string | null = null;
  lastRenderedData: VisualNetworkData | undefined;
  lastRenderedSelectedNode: VisualNode | null;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
    this.nodeWidth = 60; // or whatever default value you prefer
    this.nodeHeight = 40; // or whatever default value you prefer
    this.debouncedRender = debounce((data: VisualNetworkData | null, selectedNode: VisualNode | null) => {
      this._render(data, selectedNode);
    }, 16); // Debounce to ~60fps
    this.lastRenderedSelectedNode = null;
  }

  render(data: VisualNetworkData, selectedNode: VisualNode | null) {
    this.lastRenderedData = data;
    this.lastRenderedSelectedNode = selectedNode;
    this._render(data, selectedNode);
  }

  setHighlightedNode(nodeId: string | null) {
    this.highlightedNodeId = nodeId;
    this.debouncedRender(this.lastRenderedData, this.lastRenderedSelectedNode);
  }

  pan(dx: number, dy: number) {
    this.offsetX += dx / this.scale; // Adjust for current scale
    this.offsetY += dy / this.scale; // Adjust for current scale
  }

  zoom(x: number, y: number, factor: number) {
    const prevScale = this.scale;
    this.scale *= factor;
    this.scale = Math.max(0.1, Math.min(5, this.scale)); // Limit zoom level

    // Adjust offset to zoom towards mouse position
    this.offsetX = x - (x - this.offsetX) * (this.scale / prevScale);
    this.offsetY = y - (y - this.offsetY) * (this.scale / prevScale);
  }

  getScaledMousePosition(clientX: number, clientY: number): { x: number, y: number } {
    const rect = this.canvas.getBoundingClientRect();
    const x = (clientX - rect.left - this.offsetX) / this.scale;
    const y = (clientY - rect.top - this.offsetY) / this.scale;
    return { x, y };
  }

  private clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  private drawNodes(nodes: VisualNode[], selectedNode: VisualNode | null) {
    nodes.forEach(node => {
      const isHighlighted = node.id === this.highlightedNodeId;
      switch (node.layerId) {
        case 'input':
          this.drawInputNode(node, isHighlighted);
          break;
        default:
          this.drawHiddenNode(node, selectedNode, isHighlighted);
          break;
      }
    });
  }

  private drawHiddenNode(node: VisualNode, selectedNode: VisualNode | null, isHighlighted: boolean) {
    const nodeColor = 'rgba(255, 255, 255, 0.8)'; // White with 80% opacity
    const strokeColor = colors.border;
    const labelColor = colors.textDark; // Use a darker color for labels

    // Draw node
    this.ctx.fillStyle = nodeColor;
    this.ctx.strokeStyle = strokeColor;
    this.ctx.lineWidth = 1;
    this.ctx.beginPath();
    this.ctx.roundRect(node.x, node.y, 50, 30, 5);
    this.ctx.fill();
    this.ctx.stroke();

    // Draw activation function
    if (node.layerId !== 'input' && node.activation) {
      this.ctx.fillStyle = colors.primary;
      this.ctx.font = '10px Arial';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText(node.activation, node.x + 25, node.y + 10);
    }

    // Draw neuron label (N1, N2, N{i})
    this.ctx.fillStyle = labelColor;
    this.ctx.font = 'bold 9px Arial'; // Make the font bold for better visibility
    this.ctx.fillText(node.label, node.x + 25, node.y + 22);

    // Draw output value
    if (node.outputValue !== undefined) {
      this.drawOutputValue(node);
    }

    this.drawBias(node);

    if (selectedNode && node.id === selectedNode.id) {
      this.highlightSelectedNeuron(node);
    }

    if (isHighlighted) {
      this.ctx.shadowColor = colors.primary;
      this.ctx.shadowBlur = 15;
      this.ctx.lineWidth = 3;
      this.ctx.strokeStyle = colors.primary;
      this.ctx.stroke();
      this.ctx.shadowBlur = 0;
      this.ctx.lineWidth = 1;
    }
  }

  private drawInputNode(node: VisualNode, isHighlighted: boolean) {
    const nodeColor = colors.surface;
    const strokeColor = colors.primary;
    const labelColor = colors.text;
    const inputValueColor = colors.primary;

    const nodeWidth = 110;
    const nodeHeight = 40;
    const padding = 5;

    // Draw node
    this.ctx.fillStyle = nodeColor;
    this.ctx.strokeStyle = strokeColor;
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.roundRect(node.x, node.y, nodeWidth, nodeHeight, 10);
    this.ctx.fill();
    this.ctx.stroke();

    // Draw input icon (stylized "I" for Input)
    this.ctx.fillStyle = colors.primary;
    this.ctx.font = 'bold 18px Arial';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText('I', node.x + 15, node.y + nodeHeight / 2);

    // Draw neuron label
    this.ctx.fillStyle = labelColor;
    this.ctx.font = '12px Arial';
    this.ctx.textAlign = 'left';
    this.ctx.textBaseline = 'top';
    const labelText = this.truncateText(node.label, nodeWidth - 40, '12px Arial');
    this.ctx.fillText(labelText, node.x + 30, node.y + padding);

    // Draw input value
    if (node.outputValue !== undefined) {
      this.ctx.fillStyle = inputValueColor;
      this.ctx.font = 'bold 14px Arial';
      this.ctx.textAlign = 'right';
      this.ctx.textBaseline = 'bottom';
      const valueText = node.outputValue.toFixed(2);
      const truncatedValue = this.truncateText(valueText, nodeWidth - 35, 'bold 14px Arial');
      this.ctx.fillText(truncatedValue, node.x + nodeWidth - padding, node.y + nodeHeight - padding);
    }

    if (isHighlighted) {
      this.ctx.shadowColor = colors.primary;
      this.ctx.shadowBlur = 15;
      this.ctx.lineWidth = 4;
      this.ctx.strokeStyle = colors.primary;
      this.ctx.stroke();
      this.ctx.shadowBlur = 0;
      this.ctx.lineWidth = 2;
    }
  }

  private truncateText(text: string, maxWidth: number, font: string): string {
    this.ctx.font = font;
    let width = this.ctx.measureText(text).width;
    let ellipsis = '...';
    let truncated = text;

    if (width <= maxWidth) {
      return text;
    }

    while (width > maxWidth) {
      truncated = truncated.slice(0, -1);
      width = this.ctx.measureText(truncated + ellipsis).width;
    }

    return truncated + ellipsis;
  }

  private drawBias(node: VisualNode) {
    const biasX = node.x - 15;
    const biasY = node.y + 15;

    // Draw a small circle for the bias
    this.ctx.beginPath();
    this.ctx.arc(biasX, biasY, 8, 0, 2 * Math.PI);
    this.ctx.fillStyle = colors.background;
    this.ctx.fill();
    this.ctx.strokeStyle = colors.border;
    this.ctx.lineWidth = 1;
    this.ctx.stroke();

    // Draw the bias value
    this.ctx.fillStyle = colors.text;
    this.ctx.font = '8px Arial';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(node.bias.toFixed(2), biasX, biasY);
  }

  private drawOutputValue(node: VisualNode) {
    const outputX = node.x + 55;
    const outputY = node.y + 15;

    // Draw a small circle
    this.ctx.beginPath();
    this.ctx.arc(outputX, outputY, 10, 0, 2 * Math.PI);
    this.ctx.fillStyle = colors.background;
    this.ctx.fill();
    this.ctx.strokeStyle = colors.border;
    this.ctx.lineWidth = 1;
    this.ctx.stroke();

    // Draw the output value
    this.ctx.fillStyle = colors.text;
    this.ctx.font = '8px Arial';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(node.outputValue!.toFixed(2), outputX, outputY);
  }

  private drawConnections(connections: VisualConnection[], nodes: VisualNode[]) {
    connections.forEach(conn => {
      const fromNode = nodes.find(n => n.id === conn.from)!;
      const toNode = nodes.find(n => n.id === conn.to)!;

      const fromX = fromNode.x + 50;
      const fromY = fromNode.y + 15;
      const toX = toNode.x;
      const toY = toNode.y + 15;

      const connectionColor = this.getConnectionColor(conn.weight);
      this.drawCurvedArrow(fromX, fromY, toX, toY, connectionColor);

      // Draw weight label
      const midX = (fromX + toX) / 2;
      const midY = (fromY + toY) / 2 - 10;
      this.drawLabel(midX, midY, conn.weight.toFixed(2));
    });
  }

  private drawLabel(x: number, y: number, text: string) {
    this.ctx.font = '8px Arial';
    this.ctx.fillStyle = colors.textLight;
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(text, x, y);
  }

  private drawCurvedArrow(fromX: number, fromY: number, toX: number, toY: number, color: string) {
    const headLength = 5;
    const controlPointOffset = Math.abs(toX - fromX) * 0.2;

    this.ctx.strokeStyle = color;
    this.ctx.fillStyle = color;
    this.ctx.lineWidth = 1;

    // Draw the curved line
    this.ctx.beginPath();
    this.ctx.moveTo(fromX, fromY);
    this.ctx.bezierCurveTo(
      fromX + controlPointOffset, fromY,
      toX - controlPointOffset, toY,
      toX, toY
    );
    this.ctx.stroke();

    // Calculate the angle for the arrowhead
    const endTangentX = toX - controlPointOffset * 2;
    const endTangentY = toY;
    const angle = Math.atan2(toY - endTangentY, toX - endTangentX);

    // Draw the arrow head
    this.ctx.beginPath();
    this.ctx.moveTo(toX, toY);
    this.ctx.lineTo(toX - headLength * Math.cos(angle - Math.PI / 6), toY - headLength * Math.sin(angle - Math.PI / 6));
    this.ctx.lineTo(toX - headLength * Math.cos(angle + Math.PI / 6), toY - headLength * Math.sin(angle + Math.PI / 6));
    this.ctx.closePath();
    this.ctx.fill();
  }

  highlightSelectedNeuron(node: VisualNode) {
    this.ctx.save();
    this.ctx.strokeStyle = '#FF4500';
    this.ctx.lineWidth = 4;
    this.ctx.shadowColor = '#FF4500';
    this.ctx.shadowBlur = 15;
    this.ctx.beginPath();
    this.ctx.roundRect(node.x - 2, node.y - 2, 52, 34, 10);
    this.ctx.stroke();
    this.ctx.restore();
  }

  private getConnectionColor(weight: number): string {
    const normalizedWeight = Math.max(-1, Math.min(1, weight));
    const baseAlpha = 0.6;
    const alphaVariation = 0.3;

    if (normalizedWeight < 0) {
      const intensity = Math.round(255 * (1 + normalizedWeight));
      return `rgba(70, ${intensity}, 255, ${baseAlpha + alphaVariation * Math.abs(normalizedWeight)})`;
    } else {
      const intensity = Math.round(255 * (1 - normalizedWeight));
      return `rgba(255, ${intensity}, 70, ${baseAlpha + alphaVariation * Math.abs(normalizedWeight)})`;
    }
  }

  private drawConnection(from: Point, to: Point, weight: number) {
    const color = this.getConnectionColor(weight);

    this.ctx.beginPath();
    this.ctx.moveTo(from.x, from.y);
    this.ctx.lineTo(to.x, to.y);
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = Math.abs(weight) * 2 + 1;  // Ensure a minimum width of 1
    this.ctx.stroke();
  }

  private _render(data: VisualNetworkData, selectedNode: VisualNode | null) {
    this.clear();
    this.ctx.save();
    this.ctx.translate(this.offsetX, this.offsetY);
    this.ctx.scale(this.scale, this.scale);
    this.drawConnections(data.connections, data.nodes);
    this.drawNodes(data.nodes, selectedNode);
    this.ctx.restore();
  }

  updateDimensions(width: number, height: number) {
    this.canvas.width = width;
    this.canvas.height = height;
    this.render(this.lastRenderedData!, this.lastRenderedSelectedNode);
  }
}