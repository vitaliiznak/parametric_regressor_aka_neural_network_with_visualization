export interface Point {
  x: number;
  y: number;
}

export interface VisualNode extends Point {
  id: string;
  label: string;
  layerId: string;
  outputValue?: number;
  activation?: string;
}

export interface VisualConnection {
  from: string;
  to: string;
  weight: number;
  bias: number;
}

export interface VisualNetworkData {
  nodes: VisualNode[];
  connections: VisualConnection[];
}