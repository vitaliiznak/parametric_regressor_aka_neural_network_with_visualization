import { createStore } from "solid-js/store";
import { AppState } from "./types";
import { generateSampleData } from "./utils/dataGeneration";
import { MLP } from "./NeuralNetwork/mlp";
import { CONFIG } from "./config";
import { SerializableNetwork } from "./NeuralNetwork/types";
import { Trainer } from "./trainer";

const INITIAL_NETWORK = CONFIG.INITIAL_NETWORK;
const INITIAL_TRAINING = CONFIG.INITIAL_TRAINING;

// Action functions
function initializeTrainingData() {
  const trainingData = generateSampleData(100);
  const xs = trainingData.map(point => [point.x]);
  const ys = trainingData.map(point => point.y);
  setStore('trainingData', { xs, ys });
  console.log('Training data set:', JSON.parse(JSON.stringify(store.trainingData)));
}

function startTraining() {
  if (!store.trainingData) {
    console.error("No training data available");
    return;
  }

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
    learningRate: store.trainingConfig?.learningRate,
    epochs: store.trainingConfig?.epochs,
    batchSize: store.trainingConfig?.batchSize
  };

  // TODO: Implement training logic
}

function stopTraining() {
  setStore('isTraining', false);
}

function pauseTraining() {
  // TODO: Implement pause logic
}

function resumeTraining() {
  // TODO: Implement resume logic
}

function updateTrainingProgress(epoch: number, loss: number) {
  setStore({
    currentEpoch: epoch,
    currentLoss: loss
  });
}

function updateNetwork(network: MLP) {
  setStore('network', network);
}

function initializeTrainer() {
  if (!store.trainingData || !store.trainingConfig) {
    console.error("Training data or config not available");
    return;
  }

  const trainer = new Trainer(store.network, store.trainingConfig);
  trainer.setTrainingData(store.trainingData.xs, store.trainingData.ys);
  setStore('trainer', trainer);
}

function stepForward() {
  if (!store.trainer) {
    console.error("Trainer not initialized");
    return;
  }

  console.log("Starting forward step...");
  const result = store.trainer.stepForward();
  if (result) {
    console.log("Forward step completed. Result:", result);
    setStore('trainingResult', result);
  } else {
    console.log("Training completed");
  }
}

function stepBackward() {
  if (!store.trainer) {
    console.error("Trainer not initialized");
    return;
  }

  const result = store.trainer.stepBackward();
  if (result) {
    setStore('trainingResult', result);
  }
}

function updateWeights() {
  if (!store.trainer) {
    console.error("Trainer not initialized");
    return;
  }

  const result = store.trainer.updateWeights();
  if (result) {
    setStore('trainingResult', result);
    setStore('network', store.trainer.getNetwork());
  }
}

function simulateInput(input: number[]) {
  if (!store.currentInput) {
    alert("Please set input values first");
    return;
  }
  const output = store.network.forward(input);
  const layerOutputs = store.network.getLayerOutputs();
  setStore('simulationOutput', {
    input: store.currentInput,
    output: output.map(v => v.data),
    layerOutputs: layerOutputs
  });
}

// Initial state
const initialState: AppState = {
  network: new MLP(INITIAL_NETWORK),
  visualData: { nodes: [], connections: [] },
  trainingConfig: INITIAL_TRAINING,
  currentEpoch: 0,
  currentLoss: 0,
  isTraining: false,
  currentInput: [],
  simulationOutput: {
    input: [],
    output: [],
    layerOutputs: []
  },
  trainingResult: {
    step: 'forward', 
    data: {}
  },
  trainingData: null,
  trainer: null
};

export const [store, setStore] = createStore(initialState);

// Actions object
export const actions = {
  initializeTrainingData,
  startTraining,
  stopTraining,
  pauseTraining,
  resumeTraining,
  updateTrainingProgress,
  updateNetwork,
  stepForward,
  stepBackward,
  updateWeights,
  simulateInput
};

export const createAppStore = (initialState: AppState) => createStore(initialState);