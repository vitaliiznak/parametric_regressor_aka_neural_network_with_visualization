import { Component } from "solid-js";
import { css } from "@emotion/css";
import { FaSolidForward, FaSolidBackward, FaSolidCalculator } from 'solid-icons/fa';

interface TrainingControlButtonsProps {
  onSingleStepForward: () => void;
  onStepBackward: () => void;
  onUpdateWeights: () => void;
}

const TrainingControlButtons: Component<TrainingControlButtonsProps> = (props) => {
  const styles = {
    controlsContainer: css`
      display: flex;
      justify-content: center;
      margin-top: 1rem;
      gap: 0.5rem;
    `,
    controlButton: css`
      background-color: #3B82F6;
      color: white;
      border: none;
      border-radius: 0.25rem;
      padding: 0.5rem;
      cursor: pointer;
      transition: background-color 0.2s;
      display: flex;
      align-items: center;
      gap: 0.25rem;
      &:hover {
        background-color: #2563EB;
      }
    `,
  };

  return (
    <div class={styles.controlsContainer}>
      <button class={styles.controlButton} onClick={props.onSingleStepForward}>
        <FaSolidForward /> Forward Step
      </button>
      <button class={styles.controlButton} onClick={props.onStepBackward}>
        <FaSolidBackward /> Backward Step
      </button>
      <button class={styles.controlButton} onClick={props.onUpdateWeights}>
        <FaSolidCalculator /> Update Weights
      </button>
    </div>
  );
};

export default TrainingControlButtons;