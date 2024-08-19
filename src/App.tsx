import { Component, createEffect, createSignal } from 'solid-js';
import NetworkVisualizer from './NeuralNetworkVisualizer/NetworkVisualizer';
import TrainingControls from './TrainingControl/TrainingControls';
import NetworkConfigForm from './TrainingControl/NetworkConfigForm';
import TrainingConfigForm from './TrainingControl/TrainingConfigForm';
import { store, actions} from './store';
import LearningProcessVisualizer from './LearningProcessVisualizer/LearningProcessVisualizer';
import SimulationInputForm from './LearningProcessVisualizer/SimulationInputForm';
import FunctionVisualizer from './FunctionVisualizer';
import LegendAndTask from './LegendAndTask';


const App: Component = () => {


  createEffect(() => {
    actions.initializeTrainingData();
  });

  const simulateNetwork = () => {
    if (!store.currentInput) {
      alert("Please set input values first");
      return;
    }
    //alert(`Predicted productivity score: ${price.toFixed(2)}`);
    actions.simulateInput(store.currentInput)
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