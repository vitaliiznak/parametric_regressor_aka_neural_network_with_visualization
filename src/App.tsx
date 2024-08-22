import { Component, createEffect, createSignal } from 'solid-js';
import NetworkVisualizer from './NeuralNetworkVisualizer/NetworkVisualizer';
import TrainingControls from './TrainingControl/TrainingControls';
import NetworkConfigForm from './TrainingControl/NetworkConfigForm';
import TrainingConfigForm from './TrainingControl/TrainingConfigForm';
import { store, actions } from './store';

import SimulationInputForm from './LearningProcessVisualizer/SimulationInputForm';
import FunctionVisualizer from './FunctionVisualizer';
import LegendAndTask from './LegendAndTask';
import { css } from '@emotion/css';
import { colors } from './styles/colors';

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

  const styles = {
    mainContainer: css`
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
      
      @media (max-width: 768px) {
        padding: 10px;
      }
    `,
    contentWrapper: css`
      display: flex;
      flex-wrap: wrap;
      gap: 20px;
    `,
    leftPanel: css`
      flex: 2;
      min-width: 300px;
    `,
    rightPanel: css`
      flex: 1;
      min-width: 250px;
      
      @media (max-width: 768px) {
        width: 100%;
      }
    `,
    helpIcon: css`
      cursor: help;
      margin-left: 5px;
      color: ${colors.primary};
    `,
  };

  return (
    <div class={styles.mainContainer}>
      <LegendAndTask />
      <div class={styles.contentWrapper}>
        <div class={styles.leftPanel}>
      
          <NetworkVisualizer
            includeLossNode={false}
            onVisualizationUpdate={() => console.log("Visualization updated")}
            onSidebarToggle={handleSidebarToggle}
          />
       
   
          <FunctionVisualizer />
        </div>
        <div class={styles.rightPanel}>
       
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