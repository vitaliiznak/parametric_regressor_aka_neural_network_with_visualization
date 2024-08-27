import './styles/gloabal.css'
import { Component, createEffect, createSignal, Show } from 'solid-js';
import NetworkVisualizer from './NeuralNetworkVisualizer/NetworkVisualizer';
import FunctionVisualizer from './FunctionVisualizer';

import { actions } from './store';

import { css } from '@emotion/css';
import { colors } from './styles/colors';

import SidebarCockpit from './SidebarCockpit'
import CollapsibleSidebar from './components/CollapsibleSidebar';
import TutorialBar from './Tutorial/TutorialBar'; // Add this import

const App: Component = () => {
  createEffect(() => {
    actions.initializeTrainingData();
  });

  const [activeTab, setActiveTab] = createSignal<"network" | "function">("network");

  const styles = {
    mainContainer: css`
      display: flex;
      flex-direction: row;
      height: 100vh;
    `,
    content: css`
      display: flex;
      flex-direction: column;
      flex-grow: 1;
      overflow: hidden;
    `,
    visualizer: css`
      flex-grow: 1;
      background-color: #1B213D;
      border-radius: 4px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    `,
    tabContainer: css`
      display: flex;
      background-color: ${colors.background};
    `,
    tab: css`
      padding: 10px 20px;
      cursor: pointer;
      background-color: ${colors.surface};
      border: none;
      &.active {
        background-color: ${colors.primary};
        color: ${colors.text};
      }
    `,
  };

  return (
    <div class={styles.mainContainer}>
      <CollapsibleSidebar>
        <SidebarCockpit />
      </CollapsibleSidebar>
      <div class={styles.content}>
    
        <div class={styles.tabContainer}>
          <button
            class={`${styles.tab} ${activeTab() === "network" ? 'active' : ""}`}
            onClick={() => setActiveTab("network")}
          >
            Network
          </button>
          <button
            class={`${styles.tab} ${activeTab() === "function" ? 'active' : ""}`}
            onClick={() => setActiveTab("function")}
          >
            Function
          </button>
        </div>
        <div class={styles.visualizer}>
          <Show when={activeTab() === "network"}>
            <NetworkVisualizer
              includeLossNode={false}
              onVisualizationUpdate={() => console.log("Visualization updated")}
              onSidebarToggle={() => {/* Handle sidebar toggle */}}
              onResize={() => {/* Handle resize */}}
            />
          </Show>
          <Show when={activeTab() === "function"}>
            <FunctionVisualizer />
          </Show>
        </div>
      </div>
      <TutorialBar /> {/* Add this line to include the TutorialBar */}
    </div>
  );
};

export default App;