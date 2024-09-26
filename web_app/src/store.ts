import { createStore } from "solid-js/store";
import { AppState, BackwardStepGradientsPerConnection, NormalizationState, TrainingData, VisualNetworkData } from "./types";
import { generateSampleData } from "./utils/dataGeneration";
import { MLP } from "./NeuralNetwork/mlp";
import { CONFIG } from "./config";
import { Trainer } from "./trainer";
import { Value } from "./NeuralNetwork/value";
import { batch } from "solid-js";
import { ActivationFunction } from "./NeuralNetwork/types";
import { NormalizationMethod, computeNormalizer, normalizeData } from "./utils/dataNormalization";

const INITIAL_NETWORK = CONFIG.INITIAL_NETWORK;
const INITIAL_TRAINING = CONFIG.INITIAL_TRAINING;

const initialNormalizationState: NormalizationState = {
  method: 'none',
  normalizer: {
    mean: 0,
    std: 1,
    min: 0,
    max: 1,
  },
  normalizedData: null,
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
    isTraining: false,
    currentPhase: 'idle',
    currentEpoch: 0,
    iteration: 0,
    currentLoss: null,
    forwardStepResults: [],
    backwardStepGradients: [],
    weightUpdateResults: [],
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

  trainingRuns: [],

  networkUpdateTrigger: 0,

  normalization: initialNormalizationState,
};

export const [store, setStore] = createStore(initialState);

// Flag to prevent multiple initializations
let isInitializing = false;

// Action functions
function initializeTrainingData() {
  if (isInitializing) {
    console.warn("initializeTrainingData is already running.");
    return;
  }
  isInitializing = true;
  console.log("Action: initializeTrainingData started");

  try {
    const rawData = generateSampleData(1000, true);

    if (!rawData || !Array.isArray(rawData) || rawData.length !== 2) {
      console.error("Invalid training data:", rawData);
      return;
    }

    const [x, y] = rawData;

    // Transform DataPoints[] to TrainingData
    const trainingData: TrainingData = {
      xs: x.map(value => [value]), // Wrap each x value in an array
      ys: y,
    };

    // Validate transformed training data
    if (!trainingData.ys || !Array.isArray(trainingData.ys)) {
      console.error("Invalid training data after transformation:", trainingData);
      return;
    }

    const normalizer = computeNormalizer(trainingData.ys, store.normalization.method);
    const normalizedYs = trainingData.ys.map(y => normalizeData(y, normalizer, store.normalization.method));

    setStore({
      trainingData,
      normalization: {
        method: store.normalization.method,
        normalizer,
        normalizedData: normalizedYs,
      },
    });

  } catch (error) {
    console.error("Error in initializeTrainingData:", error);
  } finally {
    isInitializing = false;
  }
}

function setNormalizationMethod(method: NormalizationMethod) {
  if (!store.trainingData) {
    console.error("Training data not initialized.");
    return;
  }

  try {
    const normalizer = computeNormalizer(store.trainingData.ys, method);
    const normalizedYs = store.trainingData.ys.map(y => normalizeData(y, normalizer, method));

    setStore({
      normalization: {
        method,
        normalizer,
        normalizedData: normalizedYs,
      },
    });

    console.log("Action: setNormalizationMethod completed");
    console.log("Normalization updated:", store.normalization);
  } catch (error) {
    console.error("Error in setNormalizationMethod:", error);
  }
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
      weightUpdateResults: [],
      currentPhase: 'idle'
    });
  });
}

function singleStepForward() {
  console.log("Action: singleStepForward started");
  let trainerAux = store.trainer;
  if (!trainerAux) {
    trainerAux = initializeTrainer();
  }
  if (!trainerAux) {
    throw new Error("Trainer not available");
  }

  console.log("Action: Starting forward step...");
  const result = trainerAux.singleStepForward();
  const layerOutputs = trainerAux.network.getLayerOutputs();

  if (result === null) {
    batch(() => {
      setStore('trainingState', 'forwardStepResults', []);
      // Additional logic if needed
    });
  } else {
    batch(() => {
      setStore('trainingState', 'currentPhase', 'forward');
      setStore('trainingState', 'forwardStepResults', [
        ...store.trainingState.forwardStepResults,
        { input: result.input, output: result.output }
      ]);
      setStore('simulationResult', { input: result.input, output: result.output, layerOutputs: layerOutputs });
    });
  }

  console.log("Action: singleStepForward completed");
}

function calculateLoss() {
  console.log("Action: calculateLoss started");
  if (!store.trainer) {
    console.error("Trainer not initialized");
    return;
  }

  const result = store.trainer.calculateLoss();
  // console.log("After calling trainer.calculateLoss(). Result:", result);

  let currentLoss: number;
  if (result === null) {
    throw new Error("Error calculating loss");
  } else if (result instanceof Value) {
    currentLoss = result.data;
  } else {
    currentLoss = result;
  }
  console.log("Action: Current loss:", { currentLoss });

  batch(() => {
    setStore('trainingState', {
      currentPhase: 'loss',
      currentLoss: currentLoss,
      lossHistory: [...store.trainingState.lossHistory, currentLoss]
    });
  });

  console.log("Action: calculateLoss completed");
}

function stepBackward() {
  console.log("Action: stepBackward started");
  if (!store.trainer) {
    console.error("Trainer not initialized");
    return;
  }

  console.log("Action: Calling trainer.stepBackward()");
  let result: BackwardStepGradientsPerConnection[];
  try {
    result = store.trainer.stepBackwardAndGetGradientsGroupedByConnection();
  } catch (error) {
    console.error("Error in stepBackward:", error);
    return;
  }
  console.log("After calling trainer.stepBackward(). Result:", result);

  batch(() => {
    setStore('trainingState', 'currentPhase', 'backward');
    setStore('trainingState', 'backwardStepGradients', result);
  });

  console.log("Action: stepBackward completed");
}

function updateWeights() {
  console.log("Action: updateWeights started");
  batch(() => {
    if (!store.trainer || !store.trainingConfig) {
      console.error("Trainer or training configuration not initialized");
      return;
    }
    const result = store.trainer.updateWeights(store.trainingConfig.learningRate);

    setStore('trainingStepResult', result);
    setStore('trainingState', 'weightUpdateResults', result);
    setStore('network', store.trainer.network.clone());
    setStore('trainingState', 'currentPhase', 'idle');
    setStore('networkUpdateTrigger', store.networkUpdateTrigger + 1);

    console.log("Action: Weights updated successfully");

    // ** Reset Batch After Weight Update **
    store.trainer.resetBatch();
    batch(() => {
      setStore('trainingState', 'forwardStepResults', []);
      setStore('trainingState', 'backwardStepGradients', []);
      setStore('trainingState', 'currentPhase', 'idle');
    });
    console.log('Batch has been reset for the next training cycle.');
  });
}

function simulateInput(input: number[]) {
  if (!input || input.length === 0) {
    alert("Please set input values first");
    return;
  }
  const output = store.network.forward(input);
  const layerOutputs = store.network.getLayerOutputs();
  setStore({
    simulationResult: {
      input: input,
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
      inputSize,
      layers,
      activations
    });

    setStore('network', newNetwork);
    console.log("Action: Store updated with new network");

    // Reset training state
    trainingStateReset();
  });
}

// Action to set visual data
export function setVisualData(newVisualData: VisualNetworkData) {
  setStore('visualData', newVisualData);
}

// Action to reset VisualData
export function resetVisualData() {
  setStore("visualData", { nodes: [], connections: [] });
}



 /**
   * Runs training for a specified number of iterations with a given batch size.
   * 
   * @param batchSize - Number of Forward Steps per iteration.
   * @param iterations - Number of learning iterations.
   */
 function* runTrainingCyclesGenerator(batchSize: number, iterations: number): Generator<void, void, unknown> {
  setStore("trainingState", "isTraining", true);
  setStore("trainingState", "currentEpoch", 0);
  setStore("trainingState", "currentLoss", null);
  setStore("trainingState", "lossHistory", []);


  

  for (let epoch = 1; epoch <= iterations; epoch++) {
    if (!store.trainingState.isTraining) break; // Allow stopping the training

    // Perform Forward Steps based on Batch Size
    for (let batch = 0; batch < batchSize; batch++) {
      actions.singleStepForward();
      //yield; // Yield control after each batch
    }

    // Calculate Loss
    actions.calculateLoss();
    //yield;

    // Perform Backward Step
    actions.stepBackward();
    //yield;

    // Update Weights
    actions.updateWeights();
    //yield;

    // Update Training State
    setStore("trainingState", "currentEpoch", epoch);
    yield;

    console.log(`Epoch ${epoch} completed.`);
  }

  setStore("trainingState", "isTraining", false);
  console.log("Training completed.");
}

function scheduleTraining(generator: Generator<void, void, unknown>): void {
  const step = () => {
    const { value, done } = generator.next();
    if (!done) {
      requestAnimationFrame(step); // Schedule the next step
    }
  };
  requestAnimationFrame(step);
}

function runTrainingCycles(batchSize: number, iterations: number): void {
  const generator = runTrainingCyclesGenerator(batchSize, iterations);
  scheduleTraining(generator);
}

/**
 * Stops the ongoing training process.
 */
function stopTraining() {
  setStore("trainingState", "isTraining", false);
  console.log("Training stopped by user.");
}

// Export actions
export const actions = {
  initializeTrainingData,
  setNormalizationMethod,

  runTrainingCycles,

  singleStepForward,
  calculateLoss,
  stepBackward,
  updateWeights,
  simulateInput,

  trainingStateReset,
  updateNetworkConfig,
  setVisualData,
  resetVisualData,
  // Ensure that runLearningCycle is properly implemented or temporarily remove it
  // runLearningCycle,
};