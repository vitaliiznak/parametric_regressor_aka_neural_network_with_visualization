import { createStore, produce } from "solid-js/store";
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
  setStore('trainingState', produce(state => {
    state.isTraining = false;
  }));
}

function pauseTraining() {
  // TODO: Implement pause logic
}

function resumeTraining() {
  // TODO: Implement resume logic
}

function updateTrainingProgress(iteration: number, loss: number) {
  setStore('trainingState', { iteration, currentLoss: loss });
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
  console.log("Starting singleStepForward");
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
    setStore('trainingState', 'forwardStepsCount', 0);
    setStore('forwardStepResults', []);
    return;
  }

  console.log("Forward step completed. Result:", result);
  setStore('trainingResult', result);
  setStore('trainingState', 'forwardStepsCount', store.trainingState.forwardStepsCount + 1);
  setStore('forwardStepResults', [
    ...store.forwardStepResults,
    { 
      input: result.input, 
      output: result.output 
    }
  ]);
  // Update the trainingResult with the simulation input
  // Perform simulation using the simulationInput
  // Update the simulationResult in the store
  setStore('simulationResult', {
    input: result.input,
    output: result.output,
    layerOutputs: layerOutputs,
  });

  if (trainerAux.isReadyForLossCalculation()) {
    setStore('trainingState', 'forwardStepsCount', 0);
    setStore('forwardStepResults', []);
  }
  console.log("Finished singleStepForward");
}

function calculateLoss() {
  console.log("Starting calculateLoss");
  if (!store.trainer || store.trainingState.forwardStepsCount < store.trainingConfig.batchSize) {
    console.error("Cannot calculate loss");
    return;
  }

  console.log("Before calling trainer.calculateLoss()");
  const result = store.trainer.calculateLoss();
  console.log("After calling trainer.calculateLoss(). Result:", result);

  console.log("Finished calculateLoss");
}

function stepBackward() {
  console.log("Starting stepBackward");
  if (!store.trainer) {
    console.error("Trainer not initialized");
    return;
  }

  console.log("Calling trainer.stepBackward()");
  let result;
  try {
    result = store.trainer.stepBackward();
    console.log("Result from stepBackward:", result);
  } catch (error) {
    console.error("Error in stepBackward:", error);
    return;
  }

  if (result) {
    console.log("Updating store with result");
    try {
      setStore('trainingResult', result);
      console.log("Store updated successfully");
    } catch (error) {
      console.error("Error updating store:", error);
    }
  } else {
    console.log("No result from stepBackward");
  }
  console.log("Finished stepBackward");
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
    simulationResult: {
      input: store.currentInput,
      output: output.map(v => v.data),
      layerOutputs: layerOutputs
    },
    currentInput: input
  });
}

// Initial state
const initialState: AppState = {
  // Network configuration
  network: new MLP(INITIAL_NETWORK),
  visualData: { nodes: [], connections: [] },

  // Training configuration
  trainingConfig: INITIAL_TRAINING,
  trainingData: null,

  // Training state
  trainingState:{
    isTraining: false,
    currentPhase: 'idle',
    iteration: 0,
    currentLoss: null,
    forwardStepsCount: 0,
  },






  currentInput: [],
  simulationResult: {
    input: [],
    output: [],
    layerOutputs: []
  },
  trainingResult: {
    input: [],
    output: [],
    gradients: [],
    oldWeights: [],
    newWeights: [],

  },

  trainer: null,

  forwardStepResults: []
};

export const [store, setStore] = createStore(initialState);


// Replace setStore with loggedSetStore in your actions
export const actions = {
  initializeTrainingData,
  startTraining,
  stopTraining,
  pauseTraining,
  resumeTraining,
  updateTrainingProgress,
  updateNetwork,
  singleStepForward,
  calculateLoss,
  stepBackward,
  updateWeights,
  simulateInput
};

export const createAppStore = (initialState: AppState) => createStore(initialState);