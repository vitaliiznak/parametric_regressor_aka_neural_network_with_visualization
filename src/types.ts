import { MLP } from "./NeuralNetwork/mlp";
import { Trainer } from "./trainer";
import { NormalizationMethod, Normalizer } from "./utils/dataNormalization";

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

export interface NormalizationState {
  method: NormalizationMethod;
  normalizer: Normalizer;
  normalizedData: number[] | null;
}

export interface AppState {
  network: MLP;
  visualData: VisualNetworkData;

  trainingConfig: TrainingConfig;
  trainingData: TrainingData | null;

  trainingState: {
    currentPhase: string;
    iteration: number;
    currentLoss: number | null;
    forwardStepResults: ForwardStepResults[];
    backwardStepGradients: BackwardStepGradients[];
    weightUpdateResults: any[];
    lossHistory: number[];
  };

  trainingStepResult: TrainingStepResult;

  trainer: Trainer | null;

  currentInput: number[];

  simulationResult: SimulationResult;

  trainingRuns: any[];

  networkUpdateTrigger: number;

  normalization: NormalizationState;
}

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

export interface ForwardStepResults {
  input: number[];
  output: number[];
}

export interface TrainingStepResult {
  gradients: number[] | null;
  oldWeights: number[] | null;
  newWeights: number[] | null;
}