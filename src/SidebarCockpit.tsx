import { Component, createSignal, Show } from "solid-js";
import { css } from "@emotion/css";
import { colors } from "./styles/colors";
import { typography } from "./styles/typography";
import { commonStyles } from "./styles/common";
import NetworkConfigForm from "./TrainingControl/NetworkConfigForm";
import TrainingConfigForm from "./TrainingControl/TrainingConfigForm";
import TrainingControls from "./TrainingControl/TrainingControls";
import SimulationInputForm from "./LearningProcessVisualizer/SimulationInputForm";
import { store } from "./store";

const styles = {
  container: css`
    ${commonStyles.card}
    display: flex;
    flex-direction: column;
    height: 100vh;
    background-color: ${colors.surface};
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  `,
  content: css`
    flex-grow: 1;
    overflow-y: auto;
    padding: 1rem;
  `,
  header: css`
    display: flex;
    justify-content: space-between;
    align-items: center;
  `,
  title: css`
    font-size: ${typography.fontSize["2xl"]};
    font-weight: ${typography.fontWeight.bold};
    color: ${colors.text};
  `,
  tabs: css`
    display: flex;
    gap: 0.5rem;
  `,
  tab: css`
    ${commonStyles.button}
    font-size: ${typography.fontSize.sm};
    padding: 0.25rem 0.5rem;
    &.active {
      background-color: ${colors.primary};
      color: ${colors.surface};
    }
  `,
  metricsContainer: css`
    display: flex;
    gap: 1rem;
    justify-content: space-between;
    background-color: ${colors.background};
    padding: 0.5rem;
    border-radius: 4px;
  `,
  metric: css`
    display: flex;
    flex-direction: column;
    align-items: center;
  `,
  metricLabel: css`
    font-size: ${typography.fontSize.xs};
    color: ${colors.textLight};
  `,
  metricValue: css`
    font-size: ${typography.fontSize.lg};
    font-weight: ${typography.fontWeight.bold};
    color: ${colors.text};
  `,
  scrollableContent: css`
    flex-grow: 1;
    overflow-y: auto;
    padding: 1rem;

    /* Scrollbar styling */
    &::-webkit-scrollbar {
      width: 8px;
    }
    &::-webkit-scrollbar-track {
      background: ${colors.background};
      border-radius: 4px;
    }
    &::-webkit-scrollbar-thumb {
      background: ${colors.primary};
      border-radius: 4px;
    }
    &::-webkit-scrollbar-thumb:hover {
      background: ${colors.primaryDark};
    }

    /* For Firefox */
    scrollbar-width: thin;
    scrollbar-color: ${colors.primary} ${colors.background};
  `,
};

const Cockpit: Component = () => {
  const [activeTab, setActiveTab] = createSignal<"network" | "training" | "simulation">("network");

  return (
    <div class={styles.container}>
      <div class={styles.header}>
        <div class={styles.tabs}>
          <button
            class={`${styles.tab} ${activeTab() === "network" ? "active" : ""}`}
            onClick={() => setActiveTab("network")}
          >
            Network
          </button>
          <button
            class={`${styles.tab} ${activeTab() === "training" ? "active" : ""}`}
            onClick={() => setActiveTab("training")}
          >
            Training
          </button>
          <button
            class={`${styles.tab} ${activeTab() === "simulation" ? "active" : ""}`}
            onClick={() => setActiveTab("simulation")}
          >
            Simulation
          </button>
        </div>
      </div>
      <div class={styles.scrollableContent}>
        <div class={styles.metricsContainer}>
          <div class={styles.metric}>
            <span class={styles.metricLabel}>Current Loss</span>
            <span class={styles.metricValue}>{store.trainingState.currentLoss?.toFixed(4) || "N/A"}</span>
          </div>
          <div class={styles.metric}>
            <span class={styles.metricLabel}>Iteration</span>
            <span class={styles.metricValue}>{store.trainingState.iteration || 0}</span>
          </div>
          <div class={styles.metric}>
            <span class={styles.metricLabel}>Accuracy</span>
            <span class={styles.metricValue}>{"N/A"}</span>
          </div>
        </div>

        <Show when={activeTab() === "network"}>
          <NetworkConfigForm />
        </Show>
        <Show when={activeTab() === "training"}>
          <TrainingConfigForm />
          <TrainingControls onVisualizationUpdate={() => { }} />
        </Show>
        <Show when={activeTab() === "simulation"}>
          <SimulationInputForm onSimulate={() => { }} />
        </Show>
      </div>
    </div>
  );
};

export default Cockpit;