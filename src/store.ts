import { createStore, produce } from "solid-js/store";
import { AppState, TrainingConfig, TrainingData } from "./types";
import { generateSampleData } from "./utils/dataGeneration";
import { MLP } from "./NeuralNetwork/mlp";
import { CONFIG } from "./config";
import { Trainer } from "./trainer";
import { Value } from "./NeuralNetwork/value";
import { batch } from "solid-js";
import { ActivationFunction } from "./NeuralNetwork/types";

const INITIAL_NETWORK = CONFIG.INITIAL_NETWORK;
const INITIAL_TRAINING = CONFIG.INITIAL_TRAINING;

export type AppState = {
  // Network configuration
  network: MLP;
  visualData: { nodes: [], connections: [] };

  // Training configuration
  trainingConfig: TrainingConfig;
  trainingData: TrainingData;

  // Training state
  trainingState: {
    currentPhase: 'idle',
    iteration: 0,
    currentLoss: null,
    forwardStepResults: [],
    backwardStepGradients: [],
    lossHistory: [],
  };

  trainingStepResult: {
    gradients: [],
    oldWeights: [],
    newWeights: [],
  };

  trainer: Trainer;

  currentInput: [];
  simulationResult: {
    input: [],
    output: [],
    layerOutputs: []
  },
  trainingRuns: TrainingRun[]; // Add this line
};

const initialState: AppState = {
  // Network configuration
  network: new MLP(INITIAL_NETWORK),
  visualData: { nodes: [], connections: [] },

  // Training configuration
  trainingConfig: INITIAL_TRAINING,
  trainingData: null,

  // Training state
  trainingState: {
    currentPhase: 'idle',
    iteration: 0,
    currentLoss: null,
    forwardStepResults: [],
    backwardStepGradients: [],
    lossHistory: [],
  },

  trainingStepResult: {
    gradients: [],
    oldWeights: [],
    newWeights: [],
  },

  trainer: null,

  currentInput: [],
  simulationResult: {
    input: [],
    output: [],
    layerOutputs: []
  },
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

function trainingStateReset() {
  batch(() => {
    setStore('trainingState', {
      forwardStepResults: [],
      backwardStepGradients: [],
      lossHistory: [],
      currentLoss: null,
      currentPhase: 'idle'
    });
  });
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
      setStore('trainingState', 'forwardStepResults', []);
    });
    return;
  }

  console.log("Forward step completed. Result:", result);
  batch(() => {
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
      setStore('trainingState', {
        currentPhase: 'loss',
        currentLoss: currentLoss,
        lossHistory: [...store.trainingState.lossHistory, currentLoss]
      });
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
  } catch (error) {
    console.error("Error in stepBackward:", error);
    return;
  }
  console.log("After calling trainer.stepBackward(). Result:", result);

  if (result && Array.isArray(result)) {
    console.log("Updating store with result");
    try {
      console.log('here stepBackward', result)

      setStore('trainingState', 'backwardStepGradients', result);
      console.log("Store updated successfully");
    } catch (error) {
      console.error("Error updating store:", error);
    }
  } else {
    console.log("No valid result from stepBackward");
  }
  console.log("Finished stepBackward");
}

function updateWeights() {
  batch(() => {
    if (!store.trainer || !store.trainingConfig) {
      console.error("Trainer or training configuration not initialized");
      return;
    }
    const result = store.trainer.updateWeights(store.trainingConfig.learningRate);

    setStore('trainingStepResult', result);
    setStore('network', store.trainer.network);
    setStore('trainingState', 'currentPhase', 'update');
  });
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

function updateNetworkConfig(layers: number[], activations: ActivationFunction[]) {
  batch(() => {
    const inputSize = CONFIG.INITIAL_NETWORK.inputSize;

    const newNetwork = new MLP({
      inputSize: inputSize,
      layers: layers,
      activations: activations
    });

    setStore('network', newNetwork);
    console.log("Store updated with new network");

    // Reset training state
    trainingStateReset();
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
  simulateInput,
  trainingStateReset,
  updateNetworkConfig
};