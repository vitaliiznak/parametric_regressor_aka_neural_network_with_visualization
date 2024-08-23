import { Component, createEffect, createSignal, Show } from 'solid-js';
import NetworkVisualizer from './NeuralNetworkVisualizer/NetworkVisualizer';
import FunctionVisualizer from './FunctionVisualizer';
import ConfigPanel from './ConfigPanel';
import ControlPanel from './ControlPanel';
import LegendAndTask from './LegendAndTask';
import { store, actions } from './store';

import { css } from '@emotion/css';
import { colors } from './styles/colors';

const App: Component = () => {
  createEffect(() => {
    actions.initializeTrainingData();
  });

  const [activeTab, setActiveTab] = createSignal<"network" | "function">("network");

  const styles = {
    mainContainer: css`
      display: flex;
      flex-direction: column;
    `,
    header: css`
      display: flex;
      margin-bottom: 10px;
    `,
    tabContainer: css`
      display: flex;
    `,
    tab: css`
      padding: 4px 8px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
      background-color: ${colors.surface};
      color: ${colors.text};
      &.active {
        background-color: ${colors.primary};
        color: ${colors.surface};
      }
    `,
    content: css`
      display: flex;
      min-height: 0;
    `,
    visualizer: css`
      flex: 1;
 
      background-color: ${colors.surface};
      border-radius: 4px;
   
    `,
    sidebar: css`
      display: flex;
      flex-direction: column;
    `,
  };

  return (
    <div class={styles.mainContainer}>
      <div class={styles.header}>
        <LegendAndTask />
      </div>
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
      <div class={styles.content}>
        <div class={styles.visualizer}>
          <Show when={activeTab() === "network"}>
            <NetworkVisualizer
              includeLossNode={false}
              onVisualizationUpdate={() => console.log("Visualization updated")}
            />
          </Show>
          <Show when={activeTab() === "function"}>
            <FunctionVisualizer />
          </Show>
        </div>
        <div class={styles.sidebar}>
          <ConfigPanel />
          <ControlPanel />
        </div>
      </div>
    </div>
  );
};

export default App;