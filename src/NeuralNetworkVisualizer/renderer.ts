import { VisualNetworkData, VisualNode, VisualConnection } from './types';

export class NetworkRenderer {
  private ctx: CanvasRenderingContext2D;
  public scale: number = 1;
  public offsetX: number = 0;
  public offsetY: number = 0;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
    //this.ctx.scale(2, 2);  // Removed scaling for better resolution
  }

  render(data: VisualNetworkData) {
    this.clear();
    this.ctx.save();
    this.ctx.translate(this.offsetX, this.offsetY);
    this.ctx.scale(this.scale, this.scale);
    this.drawConnections(data.connections, data.nodes);
    this.drawNodes(data.nodes);
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

  private drawNodes(nodes: VisualNode[]) {
    nodes.forEach(node => {
      this.ctx.fillStyle = node.layerId === 'input' ? 'lightblue' : 'white';
      this.ctx.strokeStyle = 'black';
      this.ctx.lineWidth = 2;
      this.ctx.beginPath();
      this.ctx.rect(node.x, node.y, 60, 40);
      this.ctx.fill();
      this.ctx.stroke();

      this.ctx.fillStyle = 'black';
      this.ctx.font = '12px Arial';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText(node.label, node.x + 30, node.y + 20);

      if (node.value !== undefined) {
        this.ctx.font = '10px Arial';
        this.ctx.fillText(node.value.toFixed(2), node.x + 30, node.y + 35);
      }

      // Add activation function label for non-input nodes
      if (node.layerId !== 'input' && node.activation) {
        this.ctx.font = '8px Arial';
        this.ctx.fillText(node.activation, node.x + 30, node.y + 5);
      }
    });
  }

  private drawConnections(connections: VisualConnection[], nodes: VisualNode[]) {
    connections.forEach(conn => {
      const fromNode = nodes.find(n => n.id === conn.from)!;
      const toNode = nodes.find(n => n.id === conn.to)!;

      const fromX = fromNode.x + 60;
      const fromY = fromNode.y + 20;
      const toX = toNode.x;
      const toY = toNode.y + 20;

      this.drawArrow(fromX, fromY, toX, toY);

      // Draw weight label
      const midX = (fromX + toX) / 2;
      const midY = (fromY + toY) / 2;
      this.drawLabel(midX, midY - 10, `W: ${conn.weight.toFixed(2)}`, 'blue');

      // Draw bias label
      this.drawLabel(midX, midY + 10, `B: ${conn.weight.toFixed(2)}`, 'green');
    });
  }

  private drawArrow(fromX: number, fromY: number, toX: number, toY: number) {
    const headLen = 10;
    const angle = Math.atan2(toY - fromY, toX - fromX);
    const length = Math.sqrt(Math.pow(toX - fromX, 2) + Math.pow(toY - fromY, 2));

    this.ctx.beginPath();
    this.ctx.moveTo(fromX, fromY);
    this.ctx.lineTo(toX, toY);
    this.ctx.strokeStyle = 'black';
    this.ctx.lineWidth = 1;
    this.ctx.stroke();

    // Only draw arrowhead if there's enough space
    if (length > headLen * 2) {
      this.ctx.beginPath();
      this.ctx.moveTo(toX, toY);
      this.ctx.lineTo(toX - headLen * Math.cos(angle - Math.PI / 6), toY - headLen * Math.sin(angle - Math.PI / 6));
      this.ctx.lineTo(toX - headLen * Math.cos(angle + Math.PI / 6), toY - headLen * Math.sin(angle + Math.PI / 6));
      this.ctx.closePath();
      this.ctx.fillStyle = 'black';
      this.ctx.fill();
    }
  }

  private drawLabel(x: number, y: number, text: string, color: string) {
    this.ctx.font = '12px Arial';
    const metrics = this.ctx.measureText(text);
    const textWidth = metrics.width;
    const textHeight = 12; // Approximate height for Arial 12px

    // Draw background
    this.ctx.fillStyle = 'white';
    this.ctx.fillRect(x - textWidth / 2 - 2, y - textHeight / 2 - 2, textWidth + 4, textHeight + 4);

    // Draw text
    this.ctx.fillStyle = color;
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(text, x, y);
  }
}