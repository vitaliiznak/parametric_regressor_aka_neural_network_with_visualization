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
    forwardStepsCount: number;
  };

  // training process data
  TrainingResult: {
    gradients: number[] | null;
    oldWeights: number[] | null;
    newWeights: number[] | null;
  };


  trainingResult: TrainingResult;
  simulationResult: SimulationResult | null;
  currentInput: number[];




  trainer: Trainer | null;

  forwardStepResults: { input: number[], output: number[] }[];
}
/*
@TODO 
CHANGE TO 

export interface AppState {
  // Network configuration
  network: MLP;
  visualData: VisualNetworkData;

  // Training configuration
  trainingConfig: TrainingConfig;
  trainingData: TrainingData | null;


  // Current batch data
  currentBatch: {
    inputs: number[][];
    outputs: number[][];
    predictions: number[][];
  };

  // Simulation
  simulationInput: number[];
  simulationResult: SimulationResult | null;



  // Trainer instance
  trainer: Trainer | null;
}
*/



export interface TrainingResult {
  input: number[];
  output: number[];
  gradients: number[];
  oldWeights: number[];
  newWeights: number[];
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

