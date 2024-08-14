import { createStore } from "solid-js/store";
import { AppState } from "./types";

import { generateSampleData } from "./utils/dataGeneration";
import { MLP } from "./NeuralNetwork/mlp";
import { CONFIG } from "./config";

const INITIAL_NETWORK = CONFIG.INITIAL_NETWORK;
const INITIAL_TRAINING = CONFIG.INITIAL_TRAINING;

const initialState: AppState = {
  network: new MLP(INITIAL_NETWORK),
  trainingConfig: INITIAL_TRAINING,
  visualData: { nodes: [], connections: [] },
  simulationOutput: null,
  trainingResult: {
    step: 'forward', 
    data: {}
  },
  trainingData: null,
  currentInput: null,
  isTraining: false,
  currentEpoch: 0,
  currentLoss: 0,
};

export const [store, setStore] = createStore(initialState);


export const actions = {
  initializeTrainingData: () => {
    const trainingData = generateSampleData(100);
    const xs = trainingData.map(point => [point.x]);
    const ys = trainingData.map(point => point.y);
    setStore('trainingData', { xs, ys });
  },
  
  startTraining: () => {
    setStore('isTraining', true);
  },
  
  stopTraining: () => {
    setStore('isTraining', false);
  },
  
  updateTrainingProgress: (epoch: number, loss: number) => {
    setStore({
      currentEpoch: epoch,
      currentLoss: loss
    });
  },
  
  updateNetwork: (network: MLP) => {
    setStore('network', network);
  }
};

export const createAppStore = (initialState: AppState) => createStore(initialState);