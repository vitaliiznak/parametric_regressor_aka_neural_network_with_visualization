import { Component, createEffect, createSignal } from 'solid-js';
import NetworkVisualizer from './NeuralNetworkVisualizer/NetworkVisualizer';
import TrainingControls from './TrainingControl/TrainingControls';
import NetworkConfigForm from './TrainingControl/NetworkConfigForm';
import TrainingConfigForm from './TrainingControl/TrainingConfigForm';
import { store, setStore, actions} from './store';
import LearningProcessVisualizer from './LearningProcessVisualizer/LearningProcessVisualizer';
import SimulationInputForm from './LearningProcessVisualizer/SimulationInputForm';
import { Value } from './NeuralNetwork/value';
import FunctionVisualizer from './FunctionVisualizer';
import LegendAndTask from './LegendAndTask';


const App: Component = () => {


  const [predictedPrice, setPredictedPrice] = createSignal<number | null>(null);

  createEffect(() => {
    actions.initializeTrainingData();
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
    setStore('simulationOutput', {
      input: store.currentInput,
      output: output.map(v => v.data),
      layerOutputs: layerOutputs
    });

    console.log(`Predicted productivity score: ${price.toFixed(2)}`);
    alert(`Predicted productivity score: ${price.toFixed(2)}`);
  };

  return (
   
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

  );
};

export default App;