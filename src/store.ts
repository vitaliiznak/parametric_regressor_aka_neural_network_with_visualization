import { createStore } from "solid-js/store";
import { AppState } from "./types";
import { generateSampleData } from "./utils/dataGeneration";
import { MLP } from "./NeuralNetwork/mlp";
import { CONFIG } from "./config";
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

function updateTrainingProgress(iteration: number, loss: number) {
  setStore({
    currentIteration: iteration,
    currentLoss: loss
  });
}

function updateNetwork(network: MLP) {
  setStore('network', network);
}

function initializeTrainer() {
  if (!store.trainingData || !store.trainingConfig) {
    throw new Error("Training data or configuration not available");
  }

  const trainer = new Trainer(store.network, store.trainingConfig);
  trainer.setTrainingData(store.trainingData.xs, store.trainingData.ys);
  setStore('trainer', trainer);

  return trainer;
}

function singleStepForward() {
  setStore('forwardStepsCount', store.forwardStepsCount + 1);
  const { trainer } = store;
  let trainerAux = trainer
  if (!trainer) {
    trainerAux = initializeTrainer()
  }
  if(!trainerAux){
    throw new Error("Trainer not available");
  }

  console.log("Starting forward step...");
  const result = trainerAux.singleStepForward();
  const layerOutputs = trainerAux.network.getLayerOutputs();

  if (result === null) {
    console.log("Completed one epoch of training");
    setStore('forwardStepsCount', 0);
    setStore('forwardStepResults', []);
    return;
  }

  console.log("Forward step completed. Result:", result);
  setStore('trainingResult', result);
  setStore('forwardStepsCount', store.forwardStepsCount + 1);
  setStore('forwardStepResults', [
    ...store.forwardStepResults,
    { 
      input: Array.isArray(result.data.input) ? result.data.input : [], 
      output: Array.isArray(result.data.output) ? result.data.output : []
    }
  ]);
  // Update the trainingResult with the simulation input
  // Perform simulation using the simulationInput
  // Update the simulationOutput in the store
  setStore('simulationOutput', {
    input: result.data.input,
    output: result.data.output,
    layerOutputs: layerOutputs,
  });

  if (trainerAux.isReadyForLossCalculation()) {
    setStore('forwardStepsCount', 0);
    setStore('forwardStepResults', []);
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
    setStore('network', store.trainer.network);
  }
}

function simulateInput(input: number[]) {
  if (!store.currentInput) {
    alert("Please set input values first");
    return;
  }
  const output = store.network.forward(input);
  const layerOutputs = store.network.getLayerOutputs();
  setStore({
    simulationOutput: {
      input: store.currentInput,
      output: output.map(v => v.data),
      layerOutputs: layerOutputs
    },
    currentInput: input
  });
}

// Initial state
const initialState: AppState = {
  network: new MLP(INITIAL_NETWORK),
  visualData: { nodes: [], connections: [] },
  trainingConfig: INITIAL_TRAINING,
  currentIteration: 0,
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
  trainer: null,
  forwardStepsCount: 0,
  forwardStepResults: []
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
  singleStepForward,
  stepBackward,
  updateWeights,
  simulateInput
};

export const createAppStore = (initialState: AppState) => createStore(initialState);