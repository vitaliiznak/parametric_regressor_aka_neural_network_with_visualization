import { MLP } from "./NeuralNetwork/mlp";

export interface AppState {
    network: MLP;
    visualData: VisualNetworkData | null;
    trainingData: TrainingData | null;
    trainingConfig: TrainingConfig
    trainingResult: TrainingResult;
    currentEpoch: number;
    currentLoss: number;
    isTraining: boolean;
    simulationOutput: SimulationOutput | null;
    currentInput: number[] | null;
}

export interface TrainingData {
  xs: number[][];
  ys: number[];
}

export interface TrainingConfig {
  learningRate: number;
  epochs: number;
  batchSize: number;
}

export interface SimulationOutput {
    input: number[];
    output: number[];
    layerOutputs: number[][];
}

export interface VisualNode {
  id: string;
  label: string;
  layerId: string;
  x: number;
  y: number;
  weights: number[];
  bias: number;
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
  weights: number[];
  bias: number;
}

export interface VisualConnection {
  from: string;
  to: string;
  weight: number;
  bias: number;
}

export interface TrainingResult {
  step: 'forward' | 'loss' | 'backward' | 'update' | 'epoch';
  data: {
    input?: number[];
    output?: number[];
    loss?: number;
    gradients?: number[];
    oldWeights?: number[];
    newWeights?: number[];
    learningRate?: number;
    epoch?: number;
  };
}
