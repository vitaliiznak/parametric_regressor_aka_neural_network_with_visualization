import { Component, createEffect, createSignal } from 'solid-js';
import NetworkVisualizer from './NeuralNetworkVisualizer/NetworkVisualizer';
import TrainingControls from './TrainingControl/TrainingControls';
import NetworkConfigForm from './TrainingControl/NetworkConfigForm';
import TrainingConfigForm from './TrainingControl/TrainingConfigForm';
import { store, actions } from './store';
import LearningProcessVisualizer from './LearningProcessVisualizer/LearningProcessVisualizer';
import SimulationInputForm from './LearningProcessVisualizer/SimulationInputForm';
import FunctionVisualizer from './FunctionVisualizer';
import LegendAndTask from './LegendAndTask';
import { css } from '@emotion/css';

const App: Component = () => {
  createEffect(() => {
    actions.initializeTrainingData();
  });
  const [isSidebarOpen, setIsSidebarOpen] = createSignal(false);

  const handleSidebarToggle = (isOpen: boolean) => {
    console.log("Sidebar toggle called:", isOpen);
    setIsSidebarOpen(isOpen);
  };

  const simulateNetwork = () => {
    if (!store.currentInput) {
      alert("Please set input values first");
      return;
    }
    actions.simulateInput(store.currentInput)
  };

  const rightPanelStyle = css`
    flex: 1;
    min-width: 250px;
    max-width: 550px;
    padding: 20px;
    box-sizing: border-box;
  `;

  const mainContentStyle = css`
    transition: margin-right 0.3s ease-in-out;
    ${isSidebarOpen() ? 'margin-right: 10px;' : ''}
  `;

  return (
    <div class={mainContentStyle}>
      <LegendAndTask />
      <div style={{ display: 'flex' }}>
        <div style={{ flex: 2 }}>
          <NetworkVisualizer
            includeLossNode={false}
            onVisualizationUpdate={() => console.log("Visualization updated")}
            onSidebarToggle={handleSidebarToggle}
          />
          <LearningProcessVisualizer />
          <FunctionVisualizer />
        </div>
        <div class={rightPanelStyle}>
          <NetworkConfigForm />
          <TrainingConfigForm />
          <TrainingControls onVisualizationUpdate={() => console.log("Visualization updated")} />
          <SimulationInputForm onSimulate={simulateNetwork} />
        </div>
      </div>
    </div>
  );
};

export default App;