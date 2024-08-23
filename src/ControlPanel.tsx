import { Component } from "solid-js";
import { css } from "@emotion/css";
import { colors } from "./styles/colors";
import TrainingControls from "./TrainingControl/TrainingControls";
import SimulationInputForm from "./LearningProcessVisualizer/SimulationInputForm";

const ControlPanel: Component = () => {
  const styles = {
    container: css`
      background-color: ${colors.surface};
      border-radius: 4px;
    `,
    section: css`
      margin-bottom: 10px;
    `,
    sectionTitle: css`
      font-size: 14px;
      font-weight: bold;
      margin-bottom: 5px;
      color: ${colors.text};
    `,
  };

  return (
    <div class={styles.container}>
      <div class={styles.section}>
        <TrainingControls onVisualizationUpdate={() => console.log("Visualization updated")} />
      </div>
      <div class={styles.section}>
        <SimulationInputForm onSimulate={() => {}} />
      </div>
    </div>
  );
};

export default ControlPanel;