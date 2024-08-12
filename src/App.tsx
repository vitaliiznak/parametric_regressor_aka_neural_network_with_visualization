import { Component, createEffect } from 'solid-js';
import { createStore } from "solid-js/store";
import { MLP } from './NeuralNetwork/mlp';
import NetworkVisualizer from './NeuralNetworkVisualizer/NetworkVisualizer';
import TrainingControls from './TrainingControl/TrainingControls';
import NetworkConfigForm from './TrainingControl/NetworkConfigForm';
import TrainingConfigForm from './TrainingControl/TrainingConfigForm';
import TrainingStatus from './TrainingControl/TrainingStatus';
import { AppProvider } from "./AppContext";
import { CONFIG } from './config';
import { AppState } from './store';
import LearningProcessVisualizer from './LearningProcessVisualizer/LearningProcessVisualizer';
import InputForm from './LearningProcessVisualizer/InputForm';

import { Value } from './NeuralNetwork/value';

const INITIAL_NETWORK = CONFIG.INITIAL_NETWORK;
const INITIAL_TRAINING = CONFIG.INITIAL_TRAINING;

const App: Component = () => {
  const initialState: AppState = {
    network: new MLP(INITIAL_NETWORK),
    trainingConfig: INITIAL_TRAINING,
    visualData: { nodes: [], connections: [] },
    dotString: '',
    trainingHistory: [],
    lossValue: 0,
    trainingData: undefined,
    currentInput: undefined
  };

  const [store, setStore] = createStore<AppState>(initialState);

  const loadTrainingData = () => {
    const xs = [
      [1500, 3, 20],
      [1800, 4, 15],
      [2400, 4, 10],
      [3000, 5, 5]
    ];
    const ys = [300000, 400000, 500000, 600000];
    setStore('trainingData', { xs, ys });
  };

  createEffect(() => {
    loadTrainingData();
  });

  const simulateNetwork = () => {
    if (!store.currentInput) {
      alert("Please set input values first");
      return;
    }
    const input = store.currentInput.map(val => new Value(val));
    const output = store.network.forward(input);
    // Collect outputs from all neurons
    const layerOutputs = store.network.getLayerOutputs();
    console.log('here store layerOutputs', layerOutputs);
    // Update the simulationOutput
    console.log('here simulateNEtwork', {
      input: store.currentInput,
      output: output instanceof Value ? [output.data] : output.map(v => v.data),
      layerOutputs: layerOutputs
    })
    setStore('simulationOutput', {
      input: store.currentInput,
      output: output instanceof Value ? [output.data] : output.map(v => v.data),
      layerOutputs: layerOutputs
    });

  
    alert(`Network output: ${output instanceof Value ? [output.data] : output.map(v => v.data)}`);
  };

  return (
    <AppProvider store={[store, setStore]}>
      <div>
        <h1>Neural Network Visualizer</h1>
        <div style={{ display: 'flex' }}>
          <div style={{ flex: 2 }}>
            <NetworkVisualizer includeLossNode={false} onVisualizationUpdate={() => console.log("Visualization updated")} />
            <LearningProcessVisualizer />
          </div>
          <div style={{ flex: 1 }}>
            <NetworkConfigForm />
            <TrainingConfigForm />
            <TrainingControls onVisualizationUpdate={() => console.log("Visualization updated")} />
            <TrainingStatus />
            <InputForm />
            <button onClick={simulateNetwork}>Simulate Network</button>
            <div>
              <h2>Current Network Configuration</h2>
              <p>Layers: {store.network.layers.map(layer => layer.neurons.length).join(', ')}</p>
              <p>Activations: {store.network.activations.join(', ')}</p>
              <p>Current Loss: {store.lossValue.toFixed(4)}</p>
            </div>
          </div>
        </div>
      </div>
    </AppProvider>
  );
};

export default App;