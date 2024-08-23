import { Component, createSignal, Show } from "solid-js";
import { css } from "@emotion/css";
import { colors } from "./styles/colors";
import { typography } from "./styles/typography";
import { commonStyles } from "./styles/common";
import NetworkConfigForm from "./TrainingControl/NetworkConfigForm";
import TrainingConfigForm from "./TrainingControl/TrainingConfigForm";

const styles = {
  container: css`
    ${commonStyles.card}
  `,
  tabs: css`
    display: flex;
    gap: 0.25rem;
    margin-bottom: 0.5rem;
  `,
  tab: css`
    ${commonStyles.button}
    font-size: ${typography.fontSize.xs};
    padding: 0.125rem 0.25rem;
  `,
  configContent: css`
    max-height: 300px;
    overflow-y: auto;
  `,
};

const ConfigPanel: Component = () => {
  const [activeConfig, setActiveConfig] = createSignal<"network" | "training">("network");

  return (
    <div class={styles.container}>
      <h3 class={commonStyles.sectionTitle}>Configuration</h3>
      <div class={styles.tabs}>
        <button
          class={`${styles.tab} ${activeConfig() === "network" ? "active" : ""}`}
          onClick={() => setActiveConfig("network")}
        >
          Network
        </button>
        <button
          class={`${styles.tab} ${activeConfig() === "training" ? "active" : ""}`}
          onClick={() => setActiveConfig("training")}
        >
          Training
        </button>
      </div>
      <div class={styles.configContent}>
        <Show when={activeConfig() === "network"}>
          <NetworkConfigForm />
        </Show>
        <Show when={activeConfig() === "training"}>
          <TrainingConfigForm />
        </Show>
      </div>
    </div>
  );
};

export default ConfigPanel;