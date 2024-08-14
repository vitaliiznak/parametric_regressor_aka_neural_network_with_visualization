import { Component, createEffect, createSignal } from 'solid-js';
import { createStore } from "solid-js/store";
import { MLP } from './NeuralNetwork/mlp';
import NetworkVisualizer from './NeuralNetworkVisualizer/NetworkVisualizer';
import TrainingControls from './TrainingControl/TrainingControls';
import NetworkConfigForm from './TrainingControl/NetworkConfigForm';
import TrainingConfigForm from './TrainingControl/TrainingConfigForm';
import { AppProvider } from "./AppContext";
import { CONFIG } from './config';
import { AppState } from './store';
import LearningProcessVisualizer from './LearningProcessVisualizer/LearningProcessVisualizer';
import SimulationInputForm from './LearningProcessVisualizer/SimulationInputForm';
import { Value } from './NeuralNetwork/value';
import { generateSampleData, DataPoint } from './utils/dataGeneration';
import FunctionVisualizer from './FunctionVisualizer';
import LegendAndTask from './LegendAndTask';

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

  createEffect(() => {
    const sampleData = generateSampleData(100);
    setStore('trainingData', {
      xs: sampleData.map(point => [point.x]),
      ys: sampleData.map(point => point.y)
    });
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

    console.log(`Predicted productivity score: ${price.toFixed(2)}`);
    alert(`Predicted productivity score: ${price.toFixed(2)}`);
  };

  return (
    <AppProvider store={[store, setStore]}>
      <div>
        <h1>Neural Network Visualizer: ChatGPT Productivity Paradox</h1>
        <LegendAndTask/>
        <div style={{ display: 'flex' }}>
        
          <div style={{ flex: 2 }}>
            <NetworkVisualizer includeLossNode={false} onVisualizationUpdate={() => console.log("Visualization updated")} />
            <LearningProcessVisualizer />
            <FunctionVisualizer />
          </div>
          <div style={{ flex: 1 }}>
            <NetworkConfigForm />
            <TrainingConfigForm />
            <TrainingControls onVisualizationUpdate={() => console.log("Visualization updated")} />
            <SimulationInputForm onSimulateNetwork={simulateNetwork} />
      
          </div>
        </div>
      
      </div>
    </AppProvider>
  );
};

export default App;