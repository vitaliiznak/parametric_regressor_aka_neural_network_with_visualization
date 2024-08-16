import { createStore } from "solid-js/store";
import { AppState } from "./types";
import { generateSampleData } from "./utils/dataGeneration";
import { MLP } from "./NeuralNetwork/mlp";
import { CONFIG } from "./config";
import { SerializableNetwork } from "./NeuralNetwork/types";


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
    console.log('Training data set:', JSON.parse(JSON.stringify(store.trainingData)));
  },
  
  startTraining: () => {
    if (!store.trainingData) {
      console.error("No training data available");
      return;
    }

    const trainingWorker = new Worker(new URL('./workers/trainingWorker.ts', import.meta.url).href, { type: 'module' });

    setStore('trainingWorker', trainingWorker);
    setStore('isTraining', true);

    const serializableNetwork: SerializableNetwork = {
      inputSize: store.network.inputSize,
      layers: store.network.layers.map(layer => layer.neurons.length),
      activations: store.network.activations,
      weights: store.network.layers.map(layer => 
        layer.neurons.map(neuron => neuron.w.map(w => w.data))
      ),
      biases: store.network.layers.map(layer => 
        layer.neurons.map(neuron => neuron.b.data)
      )
    };

    const serializableConfig = {
      learningRate: store.trainingConfig.learningRate,
      epochs: store.trainingConfig.epochs,
      batchSize: store.trainingConfig.batchSize
    };

    const sendMessage = (type: string, data: any) => {
      try {
        const serializedData = type === 'trainingData' ? JSON.stringify(data) : data;
        trainingWorker.postMessage({ type, data: serializedData });
        console.log(`Successfully sent ${type} to worker`);
      } catch (error) {
        console.error(`Error sending ${type} to worker:`, error);
        console.log(`Problematic ${type} data:`, data);
      }
    };

    sendMessage('network', serializableNetwork);
    sendMessage('config', serializableConfig);

    // Send all training data at once
    sendMessage('trainingData', { 
      xs: store.trainingData.xs, 
      ys: store.trainingData.ys
    });

    trainingWorker.onmessage = (e: MessageEvent) => {
      const { type, data } = e.data;
      if (type === 'progress') {
        setStore('trainingResult', data);
      } else if (type === 'complete') {
        setStore('network', new MLP(data));
        setStore('isTraining', false);
        trainingWorker.terminate();
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
    if (!store.trainingWorker) {
      console.error("No training worker available");
      return;
    }
    store.trainingWorker.postMessage({ type: 'pause' });
  
  },

  resumeTraining: () => {
    if (!store.trainingWorker) {
      console.error("No training worker available");
      return;
    }
    store.trainingWorker.postMessage({ type: 'resume' });
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