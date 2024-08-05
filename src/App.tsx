import { Component } from 'solid-js';
import { createAppStore, AppState } from './store';
import { MLP } from './NeuralNetwork/mlp';
import NetworkVisualizer from './NeuralNetworkVisualizer/NetworkVisualizer';
import TrainingControls from './TrainingControl/TrainingControls';
import NetworkConfigForm from './TrainingControl/NetworkConfigForm';
import TrainingConfigForm from './TrainingControl/TrainingConfigForm';
import TrainingStatus from './TrainingControl/TrainingStatus';


const App: Component = () => {
  const initialState: AppState = {
    network: new MLP(1, [3, 4, 1]),
    trainingConfig: {
      learningRate: 0.01,
      epochs: 1000,
      batchSize: 1
    },
    visualData: { nodes: [], connections: [] }
  };

  const store = createAppStore(initialState);

  return (
    <div>
      <h1>Neural Network Visualizer</h1>
      <div style={{ display: 'flex' }}>
        <div style={{ flex: 2 }}>
          <NetworkVisualizer store={store} />
        </div>
        <div style={{ flex: 1 }}>
          <NetworkConfigForm store={store} />
          <TrainingConfigForm store={store} />
          <TrainingControls store={store} />
          <TrainingStatus store={store} />
        </div>
      </div>
    </div>
  );
};

export default App;