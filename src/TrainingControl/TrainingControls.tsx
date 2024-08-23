import { Component, createEffect, createSignal, Show } from "solid-js";
import { css } from "@emotion/css";
import { actions, store } from '../store';
import TrainingStepsVisualizer from './TrainingStepsVisualizer';
import TrainingStatus from "./TrainingStatus";
import { colors } from '../styles/colors';
import { typography } from '../styles/typography';
import { commonStyles } from '../styles/common';
import { FaSolidBackward, FaSolidCalculator, FaSolidForward, FaSolidWeightScale } from "solid-icons/fa";
import LossHistoryChart from "./LossHistoryChart";

const styles = {
  container: css`
    ${commonStyles.card}
    margin-top: 1rem;
  `,
  title: css`
    font-size: ${typography.fontSize.xl};
    font-weight: ${typography.fontWeight.bold};
    margin-bottom: 1rem;
    color: ${colors.text};
  `,
  controlsContainer: css`
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.5rem;
    margin-bottom: 1rem;
  `,
  controlButton: css`
    ${commonStyles.button}
    ${commonStyles.primaryButton}
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.25rem;
    font-size: ${typography.fontSize.xs};
    padding: 0.5rem 1rem; // Increase padding for larger buttons
    border-radius: 4px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  `,
  exportButton: css`
    ${commonStyles.button}
    ${commonStyles.secondaryButton}
    margin-top: 1rem;
  `,
  progressBar: css`
    width: 100%;
    height: 6px;
    background-color: ${colors.border};
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 1rem;
  `,
  progressFill: css`
    height: 100%;
    background-color: ${colors.primary};
    transition: width 300ms ease-in-out;
  `,
  trainingStepsContainer: css`
    display: flex;
    justify-content: space-between;
    margin-bottom: 1rem;
  `,
  trainingStepButton: css`
    ${commonStyles.button}
    ${commonStyles.primaryButton}
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.25rem;
    font-size: ${typography.fontSize.xs};
    padding: 0.5rem 1rem; // Increase padding for larger buttons
    border-radius: 4px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  `,
};

const TrainingControls: Component = () => {
  const [zoomRange, setZoomRange] = createSignal<[number, number]>([0, 100]);
  const [chartType, setChartType] = createSignal<'bar' | 'line'>('line');
  const [isLossCalculated, setIsLossCalculated] = createSignal(false);

  createEffect(() => {
    const { currentPhase } = store.trainingState
    if (currentPhase === 'forward' || currentPhase === 'backward') {
      setIsLossCalculated(false);
    } else if (currentPhase === 'loss') {
      setIsLossCalculated(true);
    }
  });

  const iterationProgress = () => {
    const currentIteration = store.trainingState.iteration || 0;
    const totalIterations = store.trainingConfig?.iterations || 1;
    return currentIteration / totalIterations;
  };

  const getLossColor = (loss: number) => {
    if (loss < 0.2) return colors.success;
    if (loss < 0.5) return colors.error;
    return colors.error;
  };

  const singleStepForward = () => {
    actions.singleStepForward();
    setIsLossCalculated(false);
  };

  const calculateLoss = () => {
    if (store.trainingState.forwardStepResults.length === 0) {
      console.error("No forward steps taken");
      return;
    }
    actions.calculateLoss();
    setIsLossCalculated(true);
  };

  const stepBackward = () => {
    actions.stepBackward();
  };
  const updateWeights = () => {
    actions.updateWeights();
  };

  const handleWheel = (e: WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 1.1 : 0.9;
    const [start, end] = zoomRange();
    const range = end - start;
    const newRange = range * delta;
    const center = (start + end) / 2;
    const newStart = Math.max(0, center - newRange / 2);
    const newEnd = Math.min(100, center + newRange / 2);
    setZoomRange([newStart, newEnd]);
  };

  return (
    <div class={styles.container}>
      <h2 class={styles.title}>Training Controls</h2>
      <TrainingStatus
        iteration={store.trainingState.iteration || 0}
        totalIterations={store.trainingConfig?.iterations || 0}
        currentLoss={store.trainingState.currentLoss}
        iterationProgress={iterationProgress()}
        getLossColor={getLossColor}
      />
      <div class={styles.progressBar}>
        <div class={styles.progressFill} style={{ width: `${iterationProgress() * 100}%` }}></div>
      </div>
      <TrainingStepsVisualizer
        forwardStepResults={store.trainingState.forwardStepResults}
        batchSize={store.trainingState.forwardStepResults.length}
        currentLoss={store.trainingState.currentLoss}
        backwardStepResults={store.trainingState.backwardStepGradients}
        weightUpdateResults={store.trainingStepResult}
      />

      <div class={styles.trainingStepsContainer}>
        <div>
          <button class={styles.trainingStepButton} onClick={singleStepForward}>
            <FaSolidForward /> Forward
          </button>
          <Show when={store.trainingState.forwardStepResults.length > 0}>
            <button class={styles.trainingStepButton} onClick={calculateLoss}>
              <FaSolidCalculator /> Loss
            </button>
          </Show>
        </div>
        <div>
          <Show when={isLossCalculated()}>
            <button class={styles.trainingStepButton} onClick={stepBackward}>
              <FaSolidBackward /> Backward
            </button>
          </Show>
          <Show when={store.trainingState.backwardStepGradients.length > 0}>
            <button class={styles.trainingStepButton} onClick={updateWeights}>
              <FaSolidWeightScale /> Update weights
            </button>
          </Show>
        </div>
      </div>

      <LossHistoryChart
        lossHistory={store.trainingState.lossHistory}
        trainingRuns={store.trainingRuns}
        selectedRuns={[]}
        chartType={chartType()}
        zoomRange={zoomRange()}
        maxLoss={5}
        handleWheel={handleWheel}
        onChartTypeChange={(type) => setChartType(type)}
        onZoomRangeChange={(range) => setZoomRange(range)}
      />
    </div>
  );
};

export default TrainingControls;