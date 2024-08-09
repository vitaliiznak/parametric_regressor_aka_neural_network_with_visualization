import { createStore } from "solid-js/store";
import { MLP } from "./NeuralNetwork/mlp";
import { VisualNetworkData } from "./NeuralNetworkVisualizer/types";
import { TrainingConfig, TrainingResult } from "./trainer";

export interface AppState {
  network: MLP;
  trainingConfig: TrainingConfig;
  trainingResult?: TrainingResult;
  simulationOutput?: SimulationOutput;
  visualData: VisualNetworkData;
  dotString: string;
  lossValue: number;
  trainingHistory: TrainingResult[];
  currentInput?: number[];
  trainingData?: {
    xs: number[][];
    ys: number[];
  };
}

export interface SimulationOutput {
  input: number[];
  output: number[];
  layerOutputs: number[][];
}

export const createAppStore = (initialState: AppState) => createStore(initialState);