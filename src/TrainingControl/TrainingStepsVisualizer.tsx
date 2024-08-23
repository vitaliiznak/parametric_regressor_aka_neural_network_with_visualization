import { Component, For, Show, createMemo } from "solid-js";
import { css } from "@emotion/css";
import { FaSolidArrowRight, FaSolidCalculator, FaSolidArrowLeft, FaSolidArrowDown, FaSolidLayerGroup } from 'solid-icons/fa';
import WeightUpdateStep from './WeightUpdateStep';
import { TrainingStepResult } from "../types";
import { colors } from '../styles/colors';

interface TrainingStepsVisualizerProps {
  forwardStepResults: { input: number[], output: number[] }[];
  backwardStepResults: { neuron: number, parameter: number, gradient: number }[];
  batchSize: number;
  currentLoss: number | null;
  weightUpdateResults: TrainingStepResult;
}

const TrainingStepsVisualizer: Component<TrainingStepsVisualizerProps> = (props) => {
  const MAX_VISIBLE_STEPS = 3;
  const visibleSteps = createMemo(() => {
    const startIndex = Math.max(0, props.forwardStepResults.length - MAX_VISIBLE_STEPS);
    return props.forwardStepResults.slice(startIndex);
  });

  const commonStepStyle = css`
    display: flex;
    flex-direction: column;
    align-items: center;
    background-color: ${colors.surface};
    border-radius: 0.5rem;
    padding: 0.5rem;
    width: 100px;
    height: 100px;
    justify-content: space-between;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s, box-shadow 0.2s;
    &:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
  `;

  const styles = {
    container: css`
      background-color: ${colors.background};
      border-radius: 0.5rem;
      padding: 0.75rem;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    `,
    stepsVisualization: css`
      display: flex;
      align-items: flex-start;
      justify-content: center;
      gap: 0.75rem;
      max-width: 300px;
      flex-wrap: wrap;
    `,
    forwardStepsContainer: css`
      position: relative;
      width: 120px;
      height: 140px;
    `,
    forwardStep: css`
      position: absolute;
      width: 100%;
      height: 100%;
      transition: all 0.3s ease;
      cursor: pointer;
      &:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      }
    `,
    step: commonStepStyle,
    stepIcon: css`
      font-size: 1.25rem;
      color: ${colors.primary};
    `,
    stepLabel: css`
      font-size: 0.75rem;
      font-weight: bold;
      color: ${colors.text};
      margin: 0.25rem 0;
    `,
    stepDetails: css`
      text-align: center;
      width: 100%;
      font-size: 0.625rem;
      color: ${colors.textLight};
    `,
    lossStep: css`
      ${commonStepStyle}
      background-color: ${colors.warning}22;
    `,
    backwardStep: css`
      ${commonStepStyle}
      background-color: ${colors.error}22;
    `,
    backwardStepIcon: css`
      font-size: 1.25rem;
      color: ${colors.error};
    `,
    stepCounter: css`
      position: absolute;
      top: -10px;
      left: 50%;
      transform: translateX(-50%);
      background-color: ${colors.primary};
      color: ${colors.surface};
      border-radius: 1rem;
      padding: 2px 6px;
      font-size: 0.625rem;
      font-weight: bold;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      display: flex;
      align-items: center;
      gap: 2px;
      z-index: 10;
    `,
    gradientsContainer: css`
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
      max-height: 100px;
      overflow-y: auto;
    `,
    gradientItem: css`
      display: flex;
      justify-content: space-between;
      font-size: 0.625rem;
      color: ${colors.textLight};
    `,
    gradientLabel: css`
      font-weight: bold;
      color: ${colors.text};
    `,
    gradientValue: css`
      color: ${colors.textLight};
    `,
  };

  return (
    <div class={styles.container}>
      <div class={styles.stepsVisualization}>
        <div class={styles.forwardStepsContainer}>
          {props.forwardStepResults.length > MAX_VISIBLE_STEPS && (
            <div class={styles.stepCounter}>
              <FaSolidLayerGroup />
              {props.forwardStepResults.length} Steps
            </div>
          )}
          <For each={visibleSteps()}>
            {(step, index) => (
              <div
                class={styles.forwardStep}
                style={{
                  transform: `translateY(${-3 * index()}px) translateX(${-1 * index()}px)`,
                  'z-index': MAX_VISIBLE_STEPS - index(),
                  opacity: Math.max(0.8, 1 - index() * 0.1),
                }}
              >
                <div class={styles.step}>
                  <div class={styles.stepIcon}>
                    <FaSolidArrowRight />
                  </div>
                  <div class={styles.stepLabel}>Forward</div>
                  <div class={styles.stepDetails}>
                    <div>In: {step.input.map(v => v.toFixed(1)).join(',')}</div>
                    <div>Out: {step.output.map(v => v.toFixed(2)).join(',')}</div>
                  </div>
                </div>
              </div>
            )}
          </For>
        </div>
        <Show when={props.currentLoss !== undefined && props.currentLoss !== null}>
          <div class={styles.lossStep}>
            <div class={styles.stepIcon}>
              <FaSolidCalculator />
            </div>
            <div class={styles.stepLabel}>Loss</div>
            <div class={styles.stepDetails}>
              <div>{props.currentLoss?.toFixed(4) || 'N/A'}</div>
            </div>
          </div>
        </Show>
        <Show when={props.backwardStepResults.length > 0}>
          <div class={styles.backwardStep}>
            <div class={styles.backwardStepIcon}>
              <FaSolidArrowLeft />
            </div>
            <div class={styles.stepLabel}>Backward</div>
            <div class={styles.stepDetails}>
              <div class={styles.gradientsContainer}>
                <For each={props.backwardStepResults}>
                  {(element, index) => (
                    <div class={styles.gradientItem}>
                      <span class={styles.gradientLabel}>Neuron {element.neuron}, Param {element.parameter}:</span>
                      <span class={styles.gradientValue}>{element.gradient.toFixed(2)}</span>
                    </div>
                  )}
                </For>
              </div>
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