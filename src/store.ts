import { createStore } from "solid-js/store";
import { MLP } from "./NeuralNetwork/mlp";
import { VisualNetworkData } from "./NeuralNetworkVisualizer/types";
import { TrainingConfig, TrainingResult } from "./trainer";

export interface AppState {
  network: MLP;
  trainingConfig: TrainingConfig;
  trainingResult?: TrainingResult;
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

export const createAppStore = (initialState: AppState) => createStore(initialState);