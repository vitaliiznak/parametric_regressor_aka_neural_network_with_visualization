import './styles/global.css'
import { Component, createEffect, createSignal, Show, onMount } from 'solid-js';
import NetworkVisualizer from './NeuralNetworkVisualizer/NetworkVisualizer';
import FunctionVisualizer from './FunctionVisualizer';
import { actions } from './store';
import { css } from '@emotion/css';
import { colors } from './styles/colors';
import SidebarCockpit from './SidebarCockpit'
import CollapsibleSidebar from './components/CollapsibleSidebar';
import TutorialBar from './Tutorial/TutorialBar';
import { typography } from './styles/typography';
import NormalizationSettings from "./LearningProcessVisualizer/NormalizationSettings";
import NormalizationVisualizer from "./LearningProcessVisualizer/NormalizationVisualizer";

const App: Component = () => {
  onMount(() => {
    actions.initializeTrainingData();
  });

  const [activeTab, setActiveTab] = createSignal<"network" | "function" | "combinedView">("network");

  const styles = {
    mainContainer: css`
      display: flex;
      flex-direction: column;
      height: 100vh;
    `,
    contentWrapper: css`
      display: flex;
      flex-direction: row;
      flex-grow: 1;
      overflow: hidden;
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
      min-height: 0;
    `,
    tabContainer: css`
      display: flex;
      background-color: ${colors.background};
      flex-shrink: 0;
    `,
    tab: css`
      padding: 0.5rem;
      cursor: pointer;
      background-color: ${colors.surface};
      border: none;
      font-size: ${typography.fontSize.xs};
      color: ${colors.textLight};

      &.active {
        background-color: ${colors.primary};
        color: ${colors.textLight};
      }

      @media (max-width: 600px) {
        padding: 0.25rem;
        font-size: ${typography.fontSize.xxs};
      }
    `,
    flexContainer: css`
      display: flex;
      flex-direction: column;
      height: 100vh;
    `,
    mainContent: css`
      flex-grow: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    `,
    combinedView: css`
      display: flex;
      flex-direction: row;
      height: 100%;
      background-color: ${colors.background};
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    `,
    combinedViewSection: css`
      flex: 1;
      min-width: 0;
      overflow: hidden;
      padding: 16px;
      display: flex;
      flex-direction: column;
      
      &:first-of-type {
        border-right: 2px solid ${colors.primary};
      }
    `,
    sectionTitle: css`
      font-size: 18px;
      font-weight: bold;
      margin-bottom: 12px;
      color: ${colors.text};
      flex-shrink: 0;
    `,
    visualizerContent: css`
      flex-grow: 1;
      min-height: 0;
      position: relative;
    `,
  };

  return (
    <div class={styles.flexContainer}>
      <div class={styles.mainContent}>
        <div class={styles.contentWrapper}>
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
              <button
                class={`${styles.tab} ${activeTab() === "combinedView" ? 'active' : ""}`}
                onClick={() => setActiveTab("combinedView")}
              >
                Combined View
              </button>
            </div>
            <div class={styles.visualizer}>
              <Show when={activeTab() === "network"}>
                <div class={styles.visualizerContent}>
                  <NetworkVisualizer
                    includeLossNode={false}
                    onVisualizationUpdate={() => console.log("Visualization updated")}
                    onSidebarToggle={() => {/* Handle sidebar toggle */ }}
                  />
                </div>
              </Show>
              <Show when={activeTab() === "function"}>
                <div class={styles.visualizerContent}>
                  <FunctionVisualizer />
                </div>
              </Show>
              <Show when={activeTab() === "combinedView"}>
                <div class={styles.combinedView}>
                  <div class={styles.combinedViewSection}>
                    <div class={styles.sectionTitle}>Network Visualization</div>
                    <div class={styles.visualizerContent}>
                      <NetworkVisualizer
                        includeLossNode={false}
                        onVisualizationUpdate={() => console.log("Visualization updated")}
                        onSidebarToggle={() => {/* Handle sidebar toggle */ }}
                      />
                    </div>
                  </div>
                  <div class={styles.combinedViewSection}>
                    <div class={styles.sectionTitle}>Function Visualization</div>
                    <div class={styles.visualizerContent}>
                      <FunctionVisualizer />
                    </div>
                  </div>
                </div>
              </Show>
            </div>
          </div>
        </div>
      </div>
      <TutorialBar />
    </div>
  );
};

export default App;