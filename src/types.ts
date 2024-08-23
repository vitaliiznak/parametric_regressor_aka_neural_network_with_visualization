import { MLP } from "./NeuralNetwork/mlp";
import { Trainer } from "./trainer";

export interface AppState {
  // Network configuration
  network: MLP;
  visualData: VisualNetworkData;

  // Training configuration
  trainingConfig: TrainingConfig;
  trainingData: TrainingData | null;

  // Training state
  trainingState: {
    isTraining: boolean;
    currentPhase: 'idle' | 'forward' | 'loss' | 'backward' | 'update' | 'iteration';
    iteration: number;
    currentLoss: number | null;
    forwardStepResults: Prediction[];
    backwardStepGradients: BackwardStepGradients
    lossHistory: number[];
  };

  // training process data
  trainingStepResult: TrainingStepResult

  trainer: Trainer | null;
  
  // Simulation
  currentInput: number[];
  simulationResult: SimulationResult | null;

}

export type BackwardStepGradients = { neuron: number; parameter: number; gradient: number }[];

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
  iterations: number;
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