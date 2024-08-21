import { MLP } from "./NeuralNetwork/mlp";
import { Trainer } from "./trainer";

export interface AppState {
    network: MLP;
    visualData: VisualNetworkData;
 

    simulationOutput: SimulationOutput | null;
    currentInput: number[];
    currentIteration: number;
    currentLoss: number;
    isTraining: boolean;

    trainingData: TrainingData | null;
    trainingConfig: TrainingConfig;
    trainingResult: TrainingResult;

    trainer: Trainer | null;

    forwardStepsCount: number;
    forwardStepResults: { input: number[], output: number[] }[];
}

export interface TrainingData {
  xs: number[][];
  ys: number[];
}

export interface TrainingConfig {
  learningRate: number;
  iterations: number;
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
  step: 'forward' | 'backward' | 'update' | 'iteration' | 'loss';
  data: {
    input?: number[];
    output?: number[];
    loss?: number;
    gradients?: number[];
    oldWeights?: number[];
    newWeights?: number[];
    learningRate?: number;
    iteration?: number;
    batchIndex?: number;
    stepIndex?: number;
  };
}