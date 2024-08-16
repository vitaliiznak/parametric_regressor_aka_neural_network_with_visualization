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
  trainingWorker: null,
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
    if (!store.trainingData) {
      console.error("No training data available");
      return;
    }

    const trainingWorker = new Worker(new URL('./workers/trainingWorker.ts', document.baseURI).toString(), { type: 'module' });

    setStore('trainingWorker', trainingWorker);
    setStore('isTraining', true);

    trainingWorker.postMessage({
      network: store.network.toJSON(),
      config: store.trainingConfig,
      xs: store.trainingData.xs,
      yt: store.trainingData.ys
    });

    trainingWorker.onmessage = (e: MessageEvent) => {
      if (e.data.type === 'progress') {
        setStore('trainingResult', e.data.data);
        store.network.updateFromJSON(e.data.data.network);
      } else if (e.data.type === 'complete') {
        store.network.updateFromJSON(e.data.data);
        setStore('isTraining', false);
      }
    };
  },
  
  stopTraining: () => {
    if (store.trainingWorker) {
      store.trainingWorker.terminate();
      setStore('trainingWorker', null);
    }
    setStore('isTraining', false);
  },
  
  pauseTraining: () => {
    if (store.trainingWorker) {
      store.trainingWorker.postMessage({ type: 'pause' });
    }
  },

  resumeTraining: () => {
    if (store.trainingWorker) {
      store.trainingWorker.postMessage({ type: 'resume' });
    }
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