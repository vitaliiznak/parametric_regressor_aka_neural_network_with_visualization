import { VisualNetworkData, VisualNode, VisualConnection } from './types';

export class NetworkRenderer {
  private ctx: CanvasRenderingContext2D;
  public scale: number = 1;
  public offsetX: number = 0;
  public offsetY: number = 0;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
  }

  render(data: VisualNetworkData) {
    this.clear();
    this.ctx.save();
    this.ctx.translate(this.offsetX, this.offsetY);
    this.ctx.scale(this.scale, this.scale);
    this.drawConnections(data.connections, data.nodes);
    this.drawNodes(data.nodes);
    this.drawInputConnections(data);
    this.ctx.restore();
  }

  private drawInputConnections(data: VisualNetworkData) {

    console.log('Drawing input connections');
    console.log('Input nodes:', data.nodes.filter(node => node.layerId === 'input'));
    console.log('First layer nodes:', data.nodes.filter(node => node.layerId === 'layer_0'));


    const inputNodes = data.nodes.filter(node => node.layerId === 'input');
   // const firstLayerNodes = data.nodes.filter(node => node.layerId === 'layer_0');
  
    inputNodes.forEach((inputNode, index) => {
      console.log(`Processing input node ${index}:`, inputNode);
      // Draw input node
      this.ctx.fillStyle = 'lightgreen';
      this.ctx.strokeStyle = 'black';
      this.ctx.lineWidth = 2;
      this.ctx.beginPath();
      this.ctx.rect(inputNode.x, inputNode.y, 60, 40);
      this.ctx.fill();
      this.ctx.stroke();
  
      // Draw input value
      this.ctx.fillStyle = 'black';
      this.ctx.font = '14px Arial';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText(inputNode.value?.toFixed(2) || 'Input', inputNode.x + 30, inputNode.y + 20);
  
   
    });
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

      // Draw activation function with bigger font
      if (node.layerId !== 'input' && node.activation) {
        this.ctx.fillStyle = 'black';
        this.ctx.font = '14px Arial'; // Bigger font for activation function
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(node.activation, node.x + 30, node.y + 15); // Adjust position
      }

      // Draw neuron label with smaller font
      this.ctx.fillStyle = 'black';
      this.ctx.font = '10px Arial'; // Smaller font for neuron label
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText(node.label, node.x + 30, node.y + 30); // Adjust position

      // if (node.value !== undefined) {
      //   this.ctx.font = '10px Arial';
      //   this.ctx.fillText(node.value.toFixed(2), node.x + 30, node.y + 35);
      // }

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

      console.log('conn', conn);
      this.drawLabel(midX, midY - 10, `W: ${conn.weight.toFixed(2)}`, 'blue');

      // Draw bias label
      this.drawLabel(midX, midY + 10, `B: ${conn.bias.toFixed(2)}`, 'green');
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