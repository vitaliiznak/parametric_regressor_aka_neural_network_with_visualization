import { Component, For, Show } from "solid-js";
import { css } from "@emotion/css";
import { FaSolidArrowRight, FaSolidCalculator, FaSolidArrowLeft, FaSolidArrowDown } from 'solid-icons/fa';
import WeightUpdateStep from './WeightUpdateStep';
import { TrainingStepResult } from "../types";

interface TrainingStepsVisualizerProps {
  forwardStepResults: { input: number[], output: number[] }[];
  backwardStepResults: number[];
  batchSize: number;
  currentLoss: number | null;
  weightUpdateResults: TrainingStepResult;
}

const TrainingStepsVisualizer: Component<TrainingStepsVisualizerProps> = (props) => {
  const styles = {
    container: css`
      margin-top: 1rem;
      background-color: #f3f4f6;
      border-radius: 0.5rem;
      padding: 1rem;
    `,
    stepsVisualization: css`
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      grid-gap: 1rem;
      align-items: start;
      overflow-x: auto;
      padding: 1rem 0;
    `,
    step: css`
      display: flex;
      flex-direction: column;
      align-items: center;
      background-color: white;
      border-radius: 0.25rem;
      padding: 0.75rem;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      transition: transform 0.2s, box-shadow 0.2s;
      &:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      }
    `,
    stepDetails: css`
      margin-top: 0.5rem;
      text-align: left;
      width: 100%;
      font-size: 0.75rem;
      color: #4B5563;
    `,
    stepIcon: css`
      font-size: 1.25rem;
      color: #3B82F6;
    `,
    stepLabel: css`
      font-size: 0.875rem;
      font-weight: bold;
      margin-top: 0.25rem;
    `,
    lossStep: css`
      display: flex;
      flex-direction: column;
      align-items: center;
      background-color: white;
      border-radius: 0.25rem;
      padding: 0.75rem;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      transition: transform 0.2s, box-shadow 0.2s;
      &:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      }
    `,
    backwardStep: css`
      display: flex;
      flex-direction: column;
      align-items: center;
      background-color: #ffe6e6;
      border: 2px solid #ff9999;
      border-radius: 0.25rem;
      padding: 0.75rem;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      transition: transform 0.2s, box-shadow 0.2s;
      &:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      }
    `,
    backwardStepIcon: css`
      font-size: 1.5rem;
      color: #ff4d4d;
    `,
    weightUpdateStep: css`
      display: flex;
      flex-direction: column;
      align-items: center;
      background-color: white;
      border-radius: 0.25rem;
      padding: 0.75rem;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      transition: transform 0.2s, box-shadow 0.2s;
      &:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      }
    `,
    weightUpdateStepIcon: css`
      font-size: 1.25rem;
      color: #3B82F6;
    `,
  };

  return (
    <div class={styles.container}>
      <div class={styles.stepsVisualization}>
        <For each={props.forwardStepResults}>
          {(step, index) => (
            <div class={styles.step}>
              <div class={styles.stepIcon}>
                <FaSolidArrowRight />
              </div>
              <div class={styles.stepLabel}>Step {index() + 1}</div>
              <div class={styles.stepDetails}>
                <div>Input: {step.input.map(v => v.toFixed(2)).join(', ')}</div>
                <div>Output: {step.output.map(v => v.toFixed(2)).join(', ')}</div>
              </div>
            </div>
          )}
        </For>
        <Show when={props.currentLoss !== undefined && props.currentLoss !== null}>
          <div class={styles.lossStep}>
            <div class={styles.stepIcon}>
              <FaSolidCalculator />
            </div>
            <div class={styles.stepLabel}>Loss Calculation</div>
            <div class={styles.stepDetails}>
              <div>Loss: {props.currentLoss?.toFixed(4) || 'N/A'}</div>
            </div>
          </div>
        </Show>
        <Show when={props.backwardStepResults.length > 0}>

          <div class={styles.backwardStep}>
            <div class={styles.backwardStepIcon}>
              <FaSolidArrowLeft />
            </div>
            <div class={styles.stepLabel}>Backward Step</div>
            <div class={styles.stepDetails}>
              <div>Gradients: {props.backwardStepResults.map(v => v.toFixed(4)).join(', ')}</div>
            </div>
          </div>

        </Show>

        <Show when={props.weightUpdateResults?.newWeights?.length}>
          <WeightUpdateStep
            oldWeights={props.weightUpdateResults.oldWeights ?? []}
            newWeights={props.weightUpdateResults.newWeights ?? []}
          />
        </Show>
      </div>
    </div>
  );
};

export default TrainingStepsVisualizer;