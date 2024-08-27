import { Point, VisualConnection, VisualNetworkData, VisualNode } from "../types";
import { debounce } from "@solid-primitives/scheduled";

export class NetworkRenderer {
  private ctx: CanvasRenderingContext2D;
  public scale: number = 1;
  public offsetX: number = 0;
  public offsetY: number = 0;
  private debouncedRender: (data: VisualNetworkData, selectedNode: VisualNode | null) => void;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
    this.debouncedRender = debounce((data: VisualNetworkData, selectedNode: VisualNode | null) => {
      this._render(data, selectedNode);
    }, 16); // Debounce to ~60fps
  }

  render(data: VisualNetworkData, selectedNode: VisualNode | null) {
    this.debouncedRender(data, selectedNode);
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
      switch (node.layerId) {
        case 'input':
          this.drawInputNode(node);
          break;
        // case 'output':
        //   this.drawOutputNode(node);
        //   break;
        default:
          this.drawHiddenNode(node, selectedNode);
          break;
      }


     
    });
  }

  private drawHiddenNode(node: VisualNode, selectedNode: VisualNode | null) {
    this.ctx.fillStyle =  'white';
    this.ctx.strokeStyle = 'black';
    this.ctx.lineWidth = 2;
    this.ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
    this.ctx.shadowBlur = 10;
    this.ctx.shadowOffsetX = 5;
    this.ctx.shadowOffsetY = 5;
    this.ctx.beginPath();
    this.ctx.roundRect(node.x, node.y, 60, 40, 10); // Use roundRect for rounded corners
    this.ctx.fill();
    this.ctx.stroke();
    this.ctx.shadowColor = 'transparent'; // Reset shadow

    // Draw activation function with bigger font
    if (node.layerId !== 'input' && node.activation) {
      this.ctx.fillStyle = 'black';
      this.ctx.font = '16px Arial'; // Increase font size
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText(node.activation, node.x + 30, node.y + 15);
    }

    // Draw neuron label with smaller font
    this.ctx.fillStyle = 'black';
    this.ctx.font = '12px Arial'; // Increase font size
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(node.label, node.x + 30, node.y + 30);

    // Draw output value for all nodes, including the last layer
    if (node.outputValue !== undefined) {
      this.drawOutputValue(node);
    }

    if (selectedNode && node.id === selectedNode.id) {
      this.highlightSelectedNeuron(node);
    }

    this.drawBias(node);
  }

  private drawInputNode(node: VisualNode) {
    this.ctx.fillStyle = 'lightgreen' 
    this.ctx.strokeStyle = 'black';
    this.ctx.lineWidth = 2;
    this.ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
    this.ctx.shadowBlur = 10;
    this.ctx.shadowOffsetX = 5;
    this.ctx.shadowOffsetY = 5;
    this.ctx.beginPath();
    this.ctx.rect(node.x, node.y, 84, 35); // Use roundRect for rounded corners
    this.ctx.fill();
    this.ctx.stroke();
    this.ctx.shadowColor = 'transparent'; // Reset shadow

    // Draw activation function with bigger font
    if (node.activation) {
      this.ctx.fillStyle = 'black';
      this.ctx.font = '16px Arial'; // Increase font size
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText(node.activation, node.x + 30, node.y + 15);
    }

    // Draw neuron label with smaller font
    this.ctx.fillStyle = 'black';
    this.ctx.font = '10px Arial'; // Increase font size
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(node.label, node.x + 42, node.y + 24);

    // Draw output value for all nodes, including the last layer
    if (node.outputValue !== undefined) {
      this.drawOutputValue(node);
    }
  
    this.drawBias(node);
  }

  private drawConnections(connections: VisualConnection[], nodes: VisualNode[]) {
    connections.forEach(conn => {
      const fromNode = nodes.find(n => n.id === conn.from)!;
      const toNode = nodes.find(n => n.id === conn.to)!;

      const fromX = fromNode.x + 60;
      const fromY = fromNode.y + 20;
      const toX = toNode.x;
      const toY = toNode.y + 20;

      const connectionColor = this.getConnectionColor(conn.weight);
      this.drawCurvedArrow(fromX, fromY, toX, toY, connectionColor);

      // Draw weight label
      const midX = (fromX + toX) / 2;
      const midY = (fromY + toY) / 2 - 20; // Offset the label above the curve
      this.drawLabel(midX, midY, `W: ${conn.weight.toFixed(2)}`);
    });
  }

  private drawLabel(x: number, y: number, text: string) {
    this.ctx.font = '12px Arial';
    const metrics = this.ctx.measureText(text);
    const textWidth = metrics.width;
    const textHeight = 12; // Approximate height for Arial 12px
    const padding = 4;
    const cornerRadius = 4;

    // Draw rounded rectangle background
    this.ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    this.ctx.beginPath();
    this.ctx.moveTo(x - textWidth / 2 - padding + cornerRadius, y - textHeight / 2 - padding);
    this.ctx.lineTo(x + textWidth / 2 + padding - cornerRadius, y - textHeight / 2 - padding);
    this.ctx.arcTo(x + textWidth / 2 + padding, y - textHeight / 2 - padding, x + textWidth / 2 + padding, y - textHeight / 2 - padding + cornerRadius, cornerRadius);
    this.ctx.lineTo(x + textWidth / 2 + padding, y + textHeight / 2 + padding - cornerRadius);
    this.ctx.arcTo(x + textWidth / 2 + padding, y + textHeight / 2 + padding, x + textWidth / 2 + padding - cornerRadius, y + textHeight / 2 + padding, cornerRadius);
    this.ctx.lineTo(x - textWidth / 2 - padding + cornerRadius, y + textHeight / 2 + padding);
    this.ctx.arcTo(x - textWidth / 2 - padding, y + textHeight / 2 + padding, x - textWidth / 2 - padding, y + textHeight / 2 + padding - cornerRadius, cornerRadius);
    this.ctx.lineTo(x - textWidth / 2 - padding, y - textHeight / 2 - padding + cornerRadius);
    this.ctx.arcTo(x - textWidth / 2 - padding, y - textHeight / 2 - padding, x - textWidth / 2 - padding + cornerRadius, y - textHeight / 2 - padding, cornerRadius);
    this.ctx.closePath();
    this.ctx.fill();

    // Draw text
    this.ctx.fillStyle = 'black';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(text, x, y);
  }

  private drawCurvedArrow(fromX: number, fromY: number, toX: number, toY: number, color: string) {
    const headLength = 10;
    const controlPointOffset = Math.abs(toX - fromX) * 0.2;

    this.ctx.strokeStyle = color;
    this.ctx.fillStyle = color;
    this.ctx.lineWidth = 2;

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

  private drawOutputValue(node: VisualNode) {
    const outputX = node.x + 64; // Right side of the node
    const outputY = node.y + 20; // Vertical center of the node

    // Draw a small circle
    this.ctx.beginPath();
    this.ctx.arc(outputX + 20, outputY, 20, 0, 2 * Math.PI);
    this.ctx.fillStyle = 'lightgreen';
    this.ctx.fill();
    this.ctx.strokeStyle = 'black';
    this.ctx.stroke();

    // Draw the output value
    this.ctx.fillStyle = 'black';
    this.ctx.font = '9px Arial';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(node.outputValue!.toFixed(4), outputX + 20, outputY);
  }

  highlightSelectedNeuron(node: VisualNode) {
    this.ctx.save();
    this.ctx.strokeStyle = '#FF4500';
    this.ctx.lineWidth = 4;
    this.ctx.shadowColor = '#FF4500';
    this.ctx.shadowBlur = 15;
    this.ctx.beginPath();
    this.ctx.roundRect(node.x - 2, node.y - 2, 64, 44, 10);
    this.ctx.stroke();
    this.ctx.restore();
  }

  private getConnectionColor(weight: number): string {
    const normalizedWeight = Math.max(-1, Math.min(1, weight)); // Clamp weight between -1 and 1
    let r, g, b;
    const baseAlpha = 0.6; // Base opacity for all lines
    const alphaVariation = 0.3; // Additional opacity based on weight magnitude

    if (normalizedWeight < 0) {
      // Negative weights: blue to light blue
      const t = normalizedWeight + 1; // t goes from 0 (at -1) to 1 (at 0)
      r = Math.round(70 + 100 * t);
      g = Math.round(130 + 100 * t);
      b = 255;
    } else {
      // Positive weights: light red to red
      const t = normalizedWeight; // t goes from 0 to 1
      r = 255;
      g = Math.round(130 - 100 * t);
      b = Math.round(130 - 130 * t);
    }

    const alpha = baseAlpha + alphaVariation * Math.abs(normalizedWeight);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
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

  private drawBias(node: VisualNode) {
    const biasX = node.x - 20; // Position the bias to the left of the node
    const biasY = node.y + 20; // Vertical center of the node

    // Draw a small circle for the bias
    this.ctx.beginPath();
    this.ctx.arc(biasX, biasY, 18, 0, 2 * Math.PI);
    this.ctx.fillStyle = 'lightyellow';
    this.ctx.fill();
    this.ctx.strokeStyle = 'black';
    this.ctx.stroke();

    // Draw the bias value
    this.ctx.fillStyle = 'black';
    this.ctx.font = '9px Arial';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(`b: ${node.bias.toFixed(2)}`, biasX, biasY);
  }
}