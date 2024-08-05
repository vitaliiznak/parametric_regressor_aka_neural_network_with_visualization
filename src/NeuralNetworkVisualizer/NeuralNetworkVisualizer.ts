export const NODE_WIDTH = 80;
export const NODE_HEIGHT = 40;
export const LAYER_SPACING = 200;
export const VERTICAL_SPACING = 60;
export const CANVAS_WIDTH = 1200;
export const CANVAS_HEIGHT = 1200;


export interface Node {
  id: string;
  label: string;
  layer: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Connection {
  from: Node;
  to: Node;
  weight: number;
  bias: number;
}


export const calculateNodePositions = (nodes: Node[], canvasWidth: number, canvasHeight: number) => {
  const layers = nodes.reduce((acc, node) => {
    if (!acc[node.layer]) acc[node.layer] = [];
    acc[node.layer].push(node);
    return acc;
  }, {} as Record<number, Node[]>);

  const layerCount = Object.keys(layers).length;

  Object.entries(layers).forEach(([layerIndex, layerNodes]) => {
    const x = parseInt(layerIndex) * LAYER_SPACING + (canvasWidth - (layerCount - 1) * LAYER_SPACING) / 2;
    const layerHeight = layerNodes.length * NODE_HEIGHT + (layerNodes.length - 1) * VERTICAL_SPACING;
    const startY = (canvasHeight - layerHeight) / 2;

    layerNodes.forEach((node, index) => {
      node.x = x - NODE_WIDTH / 2;
      node.y = startY + index * (NODE_HEIGHT + VERTICAL_SPACING);
    });
  });
};

export const drawArrow = (ctx: CanvasRenderingContext2D, fromX: number, fromY: number, toX: number, toY: number, label: string, isWeight: boolean) => {
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.stroke();

  const angle = Math.atan2(toY - fromY, toX - fromX);
  const headLen = 10;
  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX - headLen * Math.cos(angle - Math.PI / 6), toY - headLen * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(toX - headLen * Math.cos(angle + Math.PI / 6), toY - headLen * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = isWeight ? 'blue' : 'green';
  ctx.font = '12px Arial';
  const labelWidth = ctx.measureText(label).width;
  const labelX = (fromX + toX) / 2 - labelWidth / 2;
  const labelY = (fromY + toY) / 2 - 5;

  ctx.fillStyle = 'white';
  ctx.fillRect(labelX - 2, labelY - 12, labelWidth + 4, 16);
  ctx.fillStyle = isWeight ? 'blue' : 'green';
  ctx.fillText(label, labelX, labelY);
};



export default class NeuralNetworkVisualizer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private nodes: Node[];
  private connections: Connection[];
  private draggedNode: Node | null = null;
  public dosage: number = 50;

  constructor(canvas: HTMLCanvasElement, nodes: Node[], connections: Connection[]) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    this.nodes = nodes;
    this.connections = connections;

    this.init();
  }

  private init() {
    this.canvas.width = CANVAS_WIDTH;
    this.canvas.height = CANVAS_HEIGHT;
    calculateNodePositions(this.nodes, CANVAS_WIDTH, CANVAS_HEIGHT);
    this.draw();

    this.canvas.addEventListener('mousedown', this.handleMouseDown);
    this.canvas.addEventListener('mousemove', this.handleMouseMove);
    this.canvas.addEventListener('mouseup', this.handleMouseUp);
  }

  private handleMouseDown = (e: MouseEvent) => {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const clickedNode = this.nodes.find(node =>
      x >= node.x && x <= node.x + node.width &&
      y >= node.y && y <= node.y + node.height
    );

    if (clickedNode) {
      this.draggedNode = clickedNode;
    }
  };

  private handleMouseMove = (e: MouseEvent) => {
    if (this.draggedNode) {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      this.draggedNode.x = x - this.draggedNode.width / 2;
      this.draggedNode.y = y - this.draggedNode.height / 2;
      this.draw();
    }
  };

  private handleMouseUp = () => {
    this.draggedNode = null;
  };

  public updateDosage = (value: number) => {
    this.dosage = value;
    this.draw();
  };

  private drawNodes = () => {
    this.nodes.forEach(node => {
      this.ctx.strokeStyle = 'blue';
      this.ctx.strokeRect(node.x, node.y, node.width, node.height);
      this.ctx.fillStyle = 'black';
      this.ctx.fillText(node.label, node.x + 5, node.y + 25);
    });
  };

  private drawConnections = () => {
    this.connections.forEach(connection => {
      const fromX = connection.from.x + connection.from.width;
      const fromY = connection.from.y + connection.from.height / 2;
      const toX = connection.to.x;
      const toY = connection.to.y + connection.to.height / 2;

      drawArrow(this.ctx, fromX, fromY, toX, toY, `x ${connection.weight.toFixed(2)}`, true);
      drawArrow(this.ctx, (fromX + toX) / 2, (fromY + toY) / 2, toX, toY, `+ ${connection.bias.toFixed(2)}`, false);
    });
  };

  private drawLabels = () => {
    this.nodes.filter(node => node.layer > 1).forEach(outputNode => {
      const sumValue = this.connections
        .filter(conn => conn.to.id === outputNode.id)
        .reduce((sum, conn) => sum + (this.dosage * conn.weight + conn.bias), 0);
      const sumLabel = `Sum: ${sumValue.toFixed(2)}`;
      this.ctx.fillStyle = 'black';
      this.ctx.fillText(sumLabel, outputNode.x + NODE_WIDTH + 10, outputNode.y + NODE_HEIGHT / 2);
    });
  };

  private draw = () => {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.drawConnections();
    this.drawNodes();
    this.drawLabels();
  };
}