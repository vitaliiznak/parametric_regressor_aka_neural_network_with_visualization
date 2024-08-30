import { MLP } from "./NeuralNetwork/mlp";
import { Trainer } from "./trainer";


export type AppState = {
  // Network configuration
  network: MLP;
  visualData: { nodes: [], connections: [] };

  // Training configuration
  trainingConfig: TrainingConfig;
  trainingData: TrainingData | null;

  // Training state
  trainingState: {
    currentPhase: 'idle',
    iteration: 0,
    currentLoss: null,
    forwardStepResults: [],
    backwardStepGradients: [],
    lossHistory: [],
  };

  trainingStepResult: {
    gradients: [],
    oldWeights: [],
    newWeights: [],
  };

  trainer: Trainer | null;

  currentInput: [];
  simulationResult: {
    input: [],
    output: [],
    layerOutputs: []
  },
  trainingRuns: TrainingRun[]; // Add this line
};

export type BackwardStepGradients = {
  neuron: number;
  weights: number;
  bias: number;
  gradients: number[];
}[];

export interface TrainingStepResult {
  gradients: number[] | null;
  oldWeights: number[] | null;
  newWeights: number[] | null;
}

export interface Prediction {
  input: number[];
  output: number[];
}

export interface TrainingData {
  xs: number[][];
  ys: number[];
}

export interface TrainingConfig {
  learningRate: number;
}

export interface SimulationResult {
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
  inputValues?: number[];
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