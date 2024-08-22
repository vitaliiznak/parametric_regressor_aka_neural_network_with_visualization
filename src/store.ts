import { createStore, produce } from "solid-js/store";
import { AppState } from "./types";
import { generateSampleData } from "./utils/dataGeneration";
import { MLP } from "./NeuralNetwork/mlp";
import { CONFIG } from "./config";
import { Trainer } from "./trainer";
import { Value } from "./NeuralNetwork/value";
import { batch } from "solid-js";

const INITIAL_NETWORK = CONFIG.INITIAL_NETWORK;
const INITIAL_TRAINING = CONFIG.INITIAL_TRAINING;

const initialState: AppState = {
  // Network configuration
  network: new MLP(INITIAL_NETWORK),
  visualData: { nodes: [], connections: [] },


  // Training configuration
  trainingConfig: INITIAL_TRAINING,
  trainingData: null,

  // Training state
  trainingState: {
    isTraining: false,
    currentPhase: 'idle',
    iteration: 0,
    currentLoss: null,
    forwardStepsCount: 0,
    forwardStepResults: [],
    lossHistory: [],
  },
  currentInput: [],
  simulationResult: {
    input: [],
    output: [],
    layerOutputs: []
  },
  trainingResult: {
    gradients: [],
    oldWeights: [],
    newWeights: [],
  },

  trainer: null,


};


export const [store, setStore] = createStore(initialState);

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
  if (!trainerAux) {
    throw new Error("Trainer not available");
  }

  console.log("Starting forward step...");
  const result = trainerAux.singleStepForward();
  const layerOutputs = trainerAux.network.getLayerOutputs();

  if (result === null) {
    console.log("Completed one epoch of training");
    batch(() => {
      setStore('trainingState', 'forwardStepsCount', 0);
      setStore('trainingState', 'forwardStepResults', []);
    });
    return;
  }

  console.log("Forward step completed. Result:", result);
  batch(() => {
    setStore('trainingState', 'forwardStepsCount', store.trainingState.forwardStepsCount + 1);
    setStore('trainingState', 'forwardStepResults', [...store.trainingState.forwardStepResults, { input: result.input, output: result.output }]);
    setStore('simulationResult', { input: result.input, output: result.output, layerOutputs: layerOutputs });
  });

  console.log("Finished singleStepForward");
}


function calculateLoss() {
  console.log("Starting calculateLoss");
  if (!store.trainer) {
    console.error("Trainer not initialized");
    return;
  }

  const result = store.trainer.calculateLoss();
  console.log("After calling trainer.calculateLoss(). Result:", result);

  let currentLoss: number;
  if (result === null) {
    throw new Error("Error calculating loss");
  } else if (result instanceof Value) {
    currentLoss = result.data;
  } else {
    currentLoss = result;
  }
  console.log("Current loss:", {
    currentLoss
  });

  queueMicrotask(() => {
    batch(() => {
      setStore('trainingState', 'currentPhase', 'loss');
      setStore('trainingState', 'currentLoss', currentLoss);
      setStore('trainingState', 'lossHistory', [...store.trainingState.lossHistory, currentLoss]);
      setStore('trainingState', 'forwardStepsCount', 0);
      setStore('forwardStepResults', []);
    });
  });

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




// Replace setStore with loggedSetStore in your actions
export const actions = {
  initializeTrainingData,
  startTraining,
  stopTraining,
  pauseTraining,
  resumeTraining,

  singleStepForward,
  calculateLoss,
  stepBackward,
  updateWeights,
  simulateInput
};