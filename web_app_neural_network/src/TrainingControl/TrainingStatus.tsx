import { Component } from "solid-js";
import { css } from "@emotion/css";
import { colors } from "../styles/colors";

interface TrainingStatusProps {
  iteration: number;
  totalIterations: number;
  currentLoss: number | null;
  iterationProgress: number;
  getLossColor: (loss: number | null) => string;
}

const TrainingStatus: Component<TrainingStatusProps> = (props) => {
  const styles = {
    statusGrid: css`
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 1rem;
      margin-bottom: 1rem;
    `,
    statusItem: css`
      padding: 1rem;
      border-radius: 0.5rem;
      background-color: ${colors.surface};
      transition: all 0.3s ease;
      &:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
      }
    `,
    iterationItem: css`
      border-left: 4px solid ${colors.primary};
    `,
    lossItem: css`
      border-left: 4px solid ${colors.error};
    `,
    statusLabel: css`
      font-size: 0.875rem;
      color: ${colors.textLight};
      margin-bottom: 0.25rem;
    `,
    statusValue: css`
      font-size: 1.25rem;
      font-weight: bold;
    `,
    progressBar: css`
      width: 100%;
      height: 0.5rem;
      background-color: ${colors.border};
      border-radius: 0.25rem;
      overflow: hidden;
      margin-top: 0.5rem;
    `,
    progressFill: css`
      height: 100%;
      background-color: ${colors.primary};
      transition: width 300ms ease-in-out;
    `,
  };

  return (
    <div class={styles.statusGrid}>
      <div class={`${styles.statusItem} ${styles.iterationItem}`}>
        <div class={styles.statusLabel}>Iteration</div>
        <div class={styles.statusValue}>
          {props.iteration} / {props.totalIterations}
        </div>
        <div class={styles.progressBar}>
          <div class={styles.progressFill} style={{ width: `${props.iterationProgress * 100}%` }}></div>
        </div>
      </div>
      <div class={`${styles.statusItem} ${styles.lossItem}`}>
        <div class={styles.statusLabel}>Current Loss</div>
        <div class={styles.statusValue} style={{ color: props.getLossColor(props.currentLoss) }}>
          {props.currentLoss !== null ? props.currentLoss.toFixed(4) : 'N/A' }
        </div>
      </div>
    </div>
  );
};

export default TrainingStatus;