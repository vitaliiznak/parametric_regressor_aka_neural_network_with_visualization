export interface Point {
  x: number;
  y: number;
}

export interface VisualNode extends Point {
  id: string;
  label: string;
  layerId: string;
  value?: number;
  activation?: string; // Ensure this line is present
}

export interface VisualConnection {
  from: string;
  to: string;
  weight: number;
}

export interface VisualNetworkData {
  nodes: VisualNode[];
  connections: VisualConnection[];
}