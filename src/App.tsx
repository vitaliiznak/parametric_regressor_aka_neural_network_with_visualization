import { Component, ErrorBoundary } from 'solid-js';
import { createAppStore, AppState } from './store';
import { MLP } from './NeuralNetwork/mlp';
import NetworkVisualizer from './NeuralNetworkVisualizer/NetworkVisualizer';
import TrainingControls from './TrainingControl/TrainingControls';
import NetworkConfigForm from './TrainingControl/NetworkConfigForm';
import TrainingConfigForm from './TrainingControl/TrainingConfigForm';
import TrainingStatus from './TrainingControl/TrainingStatus';
import { AppProvider } from "./AppContext";
import { CONFIG } from './config';

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

  const store = createAppStore(initialState);

  return (
    // <ErrorBoundary fallback={(err, reset) => (
    //   <div>
    //     <p>Something went wrong: {err.toString()}</p>
    //     <button onClick={reset}>Try again</button>
    //   </div>
    // )}>
      <AppProvider store={store}>
        <div>
          <h1>Neural Network Visualizer</h1>

          <div style={{ display: 'flex' }}>
            <div style={{ flex: 2 }}>
              <NetworkVisualizer includeLossNode={true} />
            </div>
            <div style={{ flex: 1 }}>
              <NetworkConfigForm />
              <TrainingConfigForm />
              <TrainingControls />
              <TrainingStatus />
              <div>
                <h2>Current Network Configuration</h2>
                <p>Layers: {store.getState().network.layers.map(layer => layer.neurons.length).join(', ')}</p>
                <p>Activations: {store.getState().network.activations.join(', ')}</p>
                <p>Current Loss: {store.getState().lossValue.toFixed(4)}</p>
              </div>
            
            </div>
          </div>
        </div>
      </AppProvider>
    // </ErrorBoundary>
  );
};

export default App;