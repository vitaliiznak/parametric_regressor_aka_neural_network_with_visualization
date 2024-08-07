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

const INITIAL_NETWORK = CONFIG.INITIAL_NETWORK;
const INITIAL_TRAINING = CONFIG.INITIAL_TRAINING;

const App: Component = () => {
  const initialState: AppState = {
    network: new MLP(INITIAL_NETWORK),
    trainingConfig: INITIAL_TRAINING,
    visualData: { nodes: [], connections: [] },
    dotString: '',
    lossValue: 0
  };

  const [store, setStore] = createStore<AppState>(initialState);

  createEffect(() => {
    console.log("Current store state:", store);
  });

  return (
    <AppProvider store={[store, setStore]}>
      <div>
        <h1>Neural Network Visualizer</h1>
        <div style={{ display: 'flex' }}>
          <div style={{ flex: 2 }}>
            <NetworkVisualizer includeLossNode={true} />
            <LearningProcessVisualizer />
          </div>
          <div style={{ flex: 1 }}>
            <NetworkConfigForm />
            <TrainingConfigForm />
            <TrainingControls />
            <TrainingStatus />
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