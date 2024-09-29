import { throttle } from "@solid-primitives/scheduled";
import { colors } from "../styles/colors";
import { Point, VisualConnection, VisualNetworkData, VisualNode } from "../types";

export class NetworkRenderer {
  private ctx: CanvasRenderingContext2D;
  public scale: number = 1;
  public offsetX: number = 0;
  public offsetY: number = 0;
  private debouncedRender: (data: VisualNetworkData, selectedNode: VisualNode | null, highlightedConnectionId: string | null) => void;
  public nodeWidth: number;
  public nodeHeight: number;
  private highlightedNodeId: string | null = null;
  private lastRenderedData: VisualNetworkData | undefined;
  private lastRenderedSelectedNode: VisualNode | null = null;
  private onConnectionClick: (connection: VisualConnection) => void = () => { };
  private connectionControlPoints: { connection: VisualConnection; p0: Point; p1: Point; p2: Point; p3: Point }[] = [];
  private readonly epsilon: number = 5; // pixels
  private labelBoundingBoxes: { connection: VisualConnection; rect: { x: number; y: number; width: number; height: number } }[] = [];

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
    this.nodeWidth = 60;
    this.nodeHeight = 40;
    this.debouncedRender = throttle((data: VisualNetworkData, selectedNode: VisualNode | null, highlightedConnectionId: string | null) => {
      this._render(data, selectedNode, highlightedConnectionId);
    }, 16); // Debounce to ~60fps
  }

  render(data?: VisualNetworkData, selectedNode?: VisualNode | null) {
    if (data && selectedNode !== undefined) {
      this.lastRenderedData = data;
      this.lastRenderedSelectedNode = selectedNode;
    } else if (!this.lastRenderedData) {
      console.warn('No data available to render.');
      return;
    }
    this._render(this.lastRenderedData, this.lastRenderedSelectedNode, this.highlightedConnectionId);
  }

  setHighlightedNode(nodeId: string | null) {
    this.highlightedNodeId = nodeId;
    this.debouncedRender(this.lastRenderedData, this.lastRenderedSelectedNode, this.highlightedConnectionId);
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
    let nodeColor: string;
    switch (node.activation) {
      case 'tanh':
        nodeColor = 'rgba(0, 123, 255, 0.8)'; // Blue
        break;
      case 'relu':
        nodeColor = 'rgba(40, 167, 69, 0.8)'; // Green
        break;
      case 'leaky-relu':
        nodeColor = 'rgba(255, 193, 7, 0.8)'; // Yellow
        break;
      case 'sigmoid':
        nodeColor = 'rgba(220, 53, 69, 0.8)'; // Red
        break;
      default:
        nodeColor = 'rgba(255, 255, 255, 0.8)'; // White
    }

    const strokeColor = colors.border;
    const labelColor = colors.text;

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
    this.ctx.font = '5px Arial';
    this.ctx.fillText(node.label, node.x + 25, node.y + 22);

    // Draw bias
    this.drawBias(node);

    // Draw output value
    if (node.outputValue !== undefined) {
      this.drawOutputValue(node);
    }

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
    this.ctx.font = '18px Arial';
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
      this.ctx.font = '12px Arial';
      this.ctx.textAlign = 'right';
      this.ctx.textBaseline = 'bottom';
      const valueText = node.outputValue.toFixed(2);
      const truncatedValue = this.truncateText(valueText, nodeWidth - 35, '14px Arial');
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
    const biasX = node.x - 20;
    const biasY = node.y + 15;

    // Draw a diamond shape for the bias
    this.ctx.beginPath();
    this.ctx.moveTo(biasX, biasY - 8);
    this.ctx.lineTo(biasX + 8, biasY);
    this.ctx.lineTo(biasX, biasY + 8);
    this.ctx.lineTo(biasX - 8, biasY);
    this.ctx.closePath();
    this.ctx.fillStyle = node.bias >= 0 ? colors.secondary : colors.textLight;
    this.ctx.fill();
    this.ctx.strokeStyle = colors.border;
    this.ctx.lineWidth = 1;
    this.ctx.stroke();

    // Draw the bias value
    this.ctx.fillStyle = '#fff'; // Use white text for better contrast
    this.ctx.font = '5px Arial';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(`${node.bias >= 0 ? '+' : '-'}${Math.abs(node.bias).toFixed(2)}`, biasX, biasY); // Add + or - prefix

    // Add "Bias" label
    this.ctx.fillStyle = '#fff';
    this.ctx.font = '5px Arial';
    const biasLabel = 'Bias';

    this.ctx.fillText(biasLabel, biasX, biasY + 15);

    // **Do not store the bounding box for bias labels**
    // This ensures that clicking on bias labels does not trigger the ConnectionSidebar
  }

  private drawOutputValue(node: VisualNode) {
    const outputX = node.x + 60;
    const outputY = node.y + 15;

    // Draw a hexagon for the output value
    this.ctx.beginPath();
    this.ctx.moveTo(outputX + 10, outputY);
    this.ctx.lineTo(outputX + 5, outputY - 8.66);
    this.ctx.lineTo(outputX - 5, outputY - 8.66);
    this.ctx.lineTo(outputX - 10, outputY);
    this.ctx.lineTo(outputX - 5, outputY + 8.66);
    this.ctx.lineTo(outputX + 5, outputY + 8.66);
    this.ctx.closePath();
    this.ctx.fillStyle = colors.primary;
    this.ctx.fill();
    this.ctx.strokeStyle = colors.border;
    this.ctx.lineWidth = 1;
    this.ctx.stroke();

    // Draw the output value
    this.ctx.fillStyle = '#fff'; // Use dark text for better contrast
    this.ctx.font = '5px Arial';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(node.outputValue!.toFixed(2), outputX, outputY);

    // Add "Output" label
    this.ctx.fillStyle = colors.textLight;
    this.ctx.font = '5px Arial';
    this.ctx.fillText('Output', outputX, outputY + 15);
  }

  private drawConnections(
    connections: VisualConnection[],
    nodes: VisualNode[],
    highlightedConnectionId: string | null
  ) {
    this.labelBoundingBoxes = []; // Reset on each render
    this.connectionControlPoints = [];

    connections.forEach(conn => {
      const fromNode = nodes.find(n => n.id === conn.from);
      const toNode = nodes.find(n => n.id === conn.to);

      if (!fromNode || !toNode) {
        console.error(`Node not found for connection: ${conn.id}`);
        return; // Skip this connection
      }

      const fromX = fromNode.x + this.nodeWidth;
      const fromY = fromNode.y + this.nodeHeight / 2;
      const toX = toNode.x;
      const toY = toNode.y + this.nodeHeight / 2;

      // Define control points for Bezier curve
      const controlPointOffset = Math.abs(toX - fromX) * 0.2;
      const p0 = { x: fromX, y: fromY };
      const p1 = { x: fromX + controlPointOffset, y: fromY };
      const p2 = { x: toX - controlPointOffset, y: toY };
      const p3 = { x: toX, y: toY };

      // Set styles for connections
      if (highlightedConnectionId && highlightedConnectionId === conn.id) {
        this.ctx.lineWidth = 4;
        this.ctx.strokeStyle = colors.highlight;
        this.ctx.shadowColor = colors.highlight;
        this.ctx.shadowBlur = 15;
      } else {
        this.ctx.lineWidth = 1;
        this.ctx.strokeStyle = this.getConnectionColor(conn.weight);
        this.ctx.shadowBlur = 0;
      }

      this.drawCurvedArrow(p0.x, p0.y, p3.x, p3.y);

      // Reset shadow after drawing
      this.ctx.shadowBlur = 0;

      // Store control points for hit detection
      this.connectionControlPoints.push({ connection: conn, p0, p1, p2, p3 });

      // Calculate midpoint for labels
      const midX = (fromX + toX) / 2;
      const midY = (fromY + toY) / 2;

      // Offset for weight label
      const offsetY = -10;

      // Draw weight label
      this.drawLabel(midX, midY + offsetY, conn.weight, conn);
    });

    // Optionally, draw bias labels separately if needed
    nodes.forEach(node => {
      this.drawBias(node);
    });
  }

  private drawLabel(
    x: number,
    y: number,
    weight: number,
    connection: VisualConnection
  ) {
    this.ctx.save();
    
    // Define label dimensions
    const labelWidth = 32;  // Increased width for better text visibility
    const labelHeight = 16; // Reduced height for a more compact look
    const halfWidth = labelWidth / 2;
    const halfHeight = labelHeight / 2;

    // Set font for the text
    this.ctx.font = '5px Arial';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    
    // Format the weight value
    const formattedWeight = this.formatWeight(weight);
    
    // Choose background color based on weight
    const backgroundColor = weight >= 0 ? colors.weightPositive : colors.weightNegative;

    // Draw solid rectangle background
    this.ctx.fillStyle = backgroundColor;
    this.ctx.fillRect(x - halfWidth, y - halfHeight, labelWidth, labelHeight);

    // Add a border to the rectangle for better visibility
    this.ctx.strokeStyle = colors.border;
    this.ctx.lineWidth = 1;
    this.ctx.strokeRect(x - halfWidth, y - halfHeight, labelWidth, labelHeight);

    // Set text color
    this.ctx.fillStyle = 'white';

    // Draw the formatted weight text
    this.ctx.fillText(formattedWeight, x, y);

    this.ctx.restore();

    // Store the bounding box for hit detection
    this.labelBoundingBoxes.push({
      connection,
      rect: {
        x: x - halfWidth,
        y: y - halfHeight,
        width: labelWidth,
        height: labelHeight,
      },
    });
  }

  // Helper function to format weight values
  private formatWeight(weight: number): string {
    if (Math.abs(weight) < 0.01) {
      return weight.toExponential(1); // One decimal in exponential notation
    }
    if (Math.abs(weight) >= 10) {
      return weight.toFixed(1); // One decimal place for large numbers
    }
    return weight.toFixed(2); // Two decimal places for most numbers
  }

  private drawCurvedArrow(fromX: number, fromY: number, toX: number, toY: number) {
    const headLength = 10;
    const controlPointOffset = Math.abs(toX - fromX) * 0.2;

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
    const angle = Math.atan2(toY - fromY, toX - fromX);

    // Draw the arrowhead
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

  private _render(data: VisualNetworkData, selectedNode: VisualNode | null, highlightedConnectionId: string | null) {
    this.clear();
    this.ctx.save();
    this.ctx.translate(this.offsetX, this.offsetY);
    this.ctx.scale(this.scale, this.scale);
    this.drawConnections(data.connections, data.nodes, highlightedConnectionId);
    this.drawNodes(data.nodes, selectedNode);
    this.ctx.restore();
  }

  updateDimensions(width: number, height: number) {
    this.canvas.width = width;
    this.canvas.height = height;
    this.render(this.lastRenderedData!, this.lastRenderedSelectedNode);
  }

  setConnectionClickCallback(callback: (connection: VisualConnection) => void) {
    this.onConnectionClick = callback;
  }

  // Adjusted getConnectionAtPoint to exclude bias labels
  getConnectionAtPoint(x: number, y: number): VisualConnection | null {
    // Check proximity to connection lines
    for (const { connection, p0, p1, p2, p3 } of this.connectionControlPoints) {
      const distance = this.calculateDistanceToBezier(x, y, p0, p1, p2, p3);
      if (distance <= this.epsilon) {
        return connection;
      }
    }

    // Check if click is within any weight label bounding box
    for (const { connection, rect } of this.labelBoundingBoxes) {
      if (
        x >= rect.x &&
        x <= rect.x + rect.width &&
        y >= rect.y &&
        y <= rect.y + rect.height
      ) {
        return connection;
      }
    }

    return null;
  }

  setSelectedConnection(connection: VisualConnection | null) {
    this.selectedConnection = connection;
  }

  private calculateDistanceToBezier(
    x: number,
    y: number,
    p0: { x: number; y: number },
    p1: { x: number; y: number },
    p2: { x: number; y: number },
    p3: { x: number; y: number }
  ): number {
    const steps = 100;
    let minDist = Infinity;

    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const cx =
        Math.pow(1 - t, 3) * p0.x +
        3 * Math.pow(1 - t, 2) * t * p1.x +
        3 * (1 - t) * Math.pow(t, 2) * p2.x +
        Math.pow(t, 3) * p3.x;
      const cy =
        Math.pow(1 - t, 3) * p0.y +
        3 * Math.pow(1 - t, 2) * t * p1.y +
        3 * (1 - t) * Math.pow(t, 2) * p2.y +
        Math.pow(t, 3) * p3.y;

      const dx = cx - x;
      const dy = cy - y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < minDist) {
        minDist = dist;
      }
    }

    return minDist;
  }

  highlightConnection(connection: VisualConnection): void {
    this.highlightedConnectionId = connection.id;
    this.render(this.lastRenderedData!, this.lastRenderedSelectedNode);
  }

  clearHighlightedConnection(): void {
    this.highlightedConnectionId = null;
    this.render(this.lastRenderedData!, this.lastRenderedSelectedNode);
  }

  // Clean up when the renderer is no longer needed
  destroy() {
    // No cleanup needed
  }
}