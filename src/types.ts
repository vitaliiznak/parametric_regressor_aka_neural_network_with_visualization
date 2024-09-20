import { MLP } from "./NeuralNetwork/mlp";
import { Trainer } from "./trainer";


export type AppState = {
  // Network configuration
  network: MLP;
  visualData: VisualNetworkData;

  // Training configuration
  trainingConfig: TrainingConfig;
  trainingData: TrainingData | null;

  // Training state
  trainingState: {
    currentPhase: 'idle'| 'forward'| 'loss' | 'backward' | 'update',
    iteration: number,
    currentLoss: null | number,
    forwardStepResults: Prediction,
    backwardStepGradients: BackwardStepGradientsPerConnection[],
    weightUpdateResults: [],
    lossHistory: [],
  };

  trainingStepResult:TrainingStepResult;

  trainer: Trainer | null;

  currentInput: [];
  simulationResult: SimulationResult,
  trainingRuns: TrainingRun[]; 

  networkUpdateTrigger: number;
};

type TrainingRun = any

export interface BackwardStepGradientsPerConnection {
  connectionId: string;
  weightGradient: number;
  biasGradient: number;
}

export type BackwardStepGradients = BackwardStepGradientsPerConnection[];

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
  id: string; // Unique identifier
  from: string;
  to: string;
  weight: number;
  bias: number;
  weightGradient?: number;
  biasGradient?: number;
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
  weightGradient?: number;
  biasGradient?: number;
}