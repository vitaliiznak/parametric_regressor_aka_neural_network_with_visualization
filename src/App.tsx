import { Component, createEffect, createSignal } from 'solid-js';
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
import InputDataVisualizer from './LearningProcessVisualizer/InputDataVisualizer';

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
  const [predictedPrice, setPredictedPrice] = createSignal<number | null>(null);

  const generateSampleData = (count: number) => {
    const xs: number[][] = [];
    const ys: number[] = [];
  
    for (let i = 0; i < count; i++) {
      const size = Math.random() * 200 + 50; // 50 to 250 sq m
      const bedrooms = Math.floor(Math.random() * 4) + 1; // 1 to 4 bedrooms
      const price = size * 1000 + bedrooms * 50000 + Math.random() * 100000; // Simple price model
  
      xs.push([size, bedrooms]);
      ys.push(price);
    }
  
    return { xs, ys };
  };

  createEffect(() => {
    const sampleData = generateSampleData(100);
    setStore('trainingData', sampleData);
  });

  const simulateNetwork = () => {
    if (!store.currentInput) {
      alert("Please set input values first");
      return;
    }
    const input = store.currentInput.map(val => new Value(val));
    const output = store.network.forward(input);
    const price = output instanceof Value ? output.data : output[0].data;
    setPredictedPrice(price);
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
      output: output.map(v => v.data),
      layerOutputs: layerOutputs
    });

    console.log(`Predicted house price: $${price.toFixed(2)}`);
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
        <InputDataVisualizer />
      </div>
    </AppProvider>
  );
};

export default App;