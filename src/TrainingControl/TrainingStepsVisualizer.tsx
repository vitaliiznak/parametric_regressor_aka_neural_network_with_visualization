import { Component, For, Show, createMemo } from "solid-js";
import { css } from "@emotion/css";
import { FaSolidArrowRight, FaSolidCalculator, FaSolidArrowLeft, FaSolidArrowDown, FaSolidLayerGroup } from 'solid-icons/fa';
import WeightUpdateStep from './WeightUpdateStep';
import { TrainingStepResult } from "../types";
import { colors } from '../styles/colors';

interface TrainingStepsVisualizerProps {
  forwardStepResults: { input: number[], output: number[] }[];
  backwardStepResults: {
    neuron: number;
    weights: number;
    bias: number;
    gradients: number[];
  }[];
  currentLoss: number | null;
  weightUpdateResults: TrainingStepResult;
}

const TrainingStepsVisualizer: Component<TrainingStepsVisualizerProps> = (props) => {
  const MAX_VISIBLE_STEPS = 4;
  const visibleSteps = createMemo(() => {
    const startIndex = Math.max(0, props.forwardStepResults.length - MAX_VISIBLE_STEPS);
    return props.forwardStepResults.slice(startIndex);
  });

  const commonStepStyle = css`
    display: flex;
    flex-direction: column;
    align-items: center;
    background-color: ${colors.surface};
    border-radius: 0.75rem;
    padding: 0.75rem;
    width: 130px;
    height: 130px;
    justify-content: space-between;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s, box-shadow 0.2s;
    &:hover {
      transform: translateY(-3px);
      box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
  `;

  const styles = {
    container: css`
      background-color: ${colors.background};
      border-radius: 1rem;
      padding: 1rem;
      max-width: 400px;
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    `,
    stepsVisualization: css`
      display: flex;
      align-items: flex-start;
      justify-content: center;
      gap: 1rem;
  
      flex-wrap: wrap;
    `,
    forwardStepsContainer: css`
      position: relative;
      width: 150px;
      height: 170px;
    `,
    forwardStep: css`
      position: absolute;
      width: 100%;
      height: 100%;
      transition: all 0.3s ease;
      cursor: pointer;
      &:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
      }
    `,
    step: commonStepStyle,
    stepIcon: css`
      font-size: 1.5rem;
      color: ${colors.primary};
    `,
    stepLabel: css`
      font-size: 0.875rem;
      font-weight: bold;
      color: ${colors.text};
      margin: 0.5rem 0;
    `,
    stepDetails: css`
      text-align: center;
      width: 100%;
      font-size: 0.75rem;
      color: ${colors.textLight};
    `,
    lossStep: css`
      ${commonStepStyle}
      background-color: ${colors.warning}22;
    `,
    backwardStep: css`
      ${commonStepStyle}
      background-color: ${colors.error}11;
      width: 150px;
      height: 150px;
    `,
    backwardStepIcon: css`
      font-size: 1.5rem;
      color: ${colors.error};
    `,
    stepCounter: css`
      position: absolute;
      top: -12px;
      left: 50%;
      transform: translateX(-50%);
      background-color: ${colors.primary};
      color: ${colors.surface};
      border-radius: 1rem;
      padding: 4px 8px;
      font-size: 0.75rem;
      font-weight: bold;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      display: flex;
      align-items: center;
      gap: 4px;
      z-index: 10;
    `,
    gradientsContainer: css`
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
      max-height: 100px;
      overflow-y: auto;
      scrollbar-width: thin;
      scrollbar-color: ${colors.primary} ${colors.surface};
      &::-webkit-scrollbar {
        width: 4px;
      }
      &::-webkit-scrollbar-track {
        background: ${colors.surface};
      }
      &::-webkit-scrollbar-thumb {
        background-color: ${colors.primary};
        border-radius: 2px;
      }
    `,
    neuronGradients: css`
      background-color: ${colors.surface}88;
      border-radius: 0.25rem;
      padding: 0.25rem;
      margin-bottom: 0.25rem;
    `,
    neuronLabel: css`
      font-size: 0.75rem;
      font-weight: bold;
      color: ${colors.primary};
      margin-bottom: 0.25rem;
    `,
    gradientGroup: css`
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 0.125rem;
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
          {props.forwardStepResults.length > 1 && (
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
                  transform: `translateY(${-5 * index()}px) translateX(${-2 * index()}px)`,
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
                    <div>In: {step.input.map(v => v.toFixed(2)).join(',')}</div>
                    <div>Out: {step.output.map(v => v.toFixed(3)).join(',')}</div>
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
              <div>{props.currentLoss?.toFixed(5) || 'N/A'}</div>
            </div>
          </div>
        </Show>
        <Show when={props.currentLoss !== undefined && props.currentLoss !== null && props.backwardStepResults.length > 0}>
          <div class={styles.backwardStep}>
            <div class={styles.backwardStepIcon}>
              <FaSolidArrowLeft />
            </div>
            <div class={styles.stepLabel}>Backward</div>
            <div class={styles.stepDetails}>
              <div class={styles.gradientsContainer}>
                <For each={props.backwardStepResults}>
                  {(element, neuronIndex) => (
                    <div class={styles.neuronGradients}>
                      <h4 class={styles.neuronLabel}>Neuron {neuronIndex()}</h4>
                      <div class={styles.gradientGroup}>
                        <For each={element.gradients.slice(0, element.weights)}>
                          {(gradient, weightIndex) => (
                            <div class={styles.gradientItem}>
                              <span class={styles.gradientLabel}>W{weightIndex() + 1}:</span>
                              <span class={styles.gradientValue}>{gradient.toFixed(4)}</span>
                            </div>
                          )}
                        </For>
                        <div class={styles.gradientItem}>
                          <span class={styles.gradientLabel}>B:</span>
                          <span class={styles.gradientValue}>{element.gradients[element.gradients.length - 1].toFixed(4)}</span>
                        </div>
                      </div>
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