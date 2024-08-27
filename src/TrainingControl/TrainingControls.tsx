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
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
    margin-bottom: 1rem;
  `,
  controlButton: css`
    ${commonStyles.button}
    ${commonStyles.primaryButton}
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    font-size: ${typography.fontSize.sm};
    padding: 0.75rem 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    transition: all 0.2s ease-in-out;
    
    &:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
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
  disabledButton: css`
    opacity: 0.5;
    cursor: not-allowed;
  `,
  buttonGroup: css`
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  `,
  separator: css`
    height: 1px;
    background-color: ${colors.border};
    margin: 1rem 0;
  `,
  resetButton: css`
    ${commonStyles.button}
    ${commonStyles.secondaryButton}
    width: 100%;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    font-size: ${typography.fontSize.sm};
    transition: all 0.2s ease-in-out;
    
    &:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
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

  const isForwardDisabled = () =>
    (store.trainingState.currentPhase !== 'idle' && store.trainingState.currentPhase !== 'update') ||
    store.trainingState.backwardStepGradients.length > 0;

  const isLossDisabled = () =>
    store.trainingState.forwardStepResults.length === 0 ||
    store.trainingState.backwardStepGradients.length > 0;

  const isBackwardDisabled = () => store.trainingState.currentPhase !== 'loss';

  const isUpdateWeightsDisabled = () => store.trainingState.backwardStepGradients.length === 0;

  const isResetDisabled = () => store.trainingState.forwardStepResults.length === 0;

  const singleStepForward = () => {
    if (!isForwardDisabled()) actions.singleStepForward();
    setIsLossCalculated(false);
  };

  const trainingStateReset = () => {
    if (!isResetDisabled()) actions.trainingStateReset();
    setIsLossCalculated(false);
  };

  const calculateLoss = () => {
    if (!isLossDisabled()) actions.calculateLoss();
    setIsLossCalculated(true);
  };

  const stepBackward = () => {
    if (!isBackwardDisabled()) actions.stepBackward();
  };
  const updateWeights = () => {
    if (!isUpdateWeightsDisabled()) actions.updateWeights();
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
      <TrainingStatus
        iteration={store.trainingState.iteration || 0}
        currentLoss={store.trainingState.currentLoss}
        iterationProgress={iterationProgress()}
        getLossColor={getLossColor}
      />
      <div class={styles.progressBar}>
        <div class={styles.progressFill} style={{ width: `${iterationProgress() * 100}%` }}></div>
      </div>
      <TrainingStepsVisualizer
        forwardStepResults={store.trainingState.forwardStepResults}
        backwardStepResults={store.trainingState.backwardStepGradients}
        currentLoss={store.trainingState.currentLoss}
        weightUpdateResults={store.trainingState.weightUpdateResults}
      />
      <div class={styles.controlsContainer}>
        <div class={styles.buttonGroup}>
          <button
            class={`${styles.controlButton} ${isForwardDisabled() ? styles.disabledButton : ''}`}
            onClick={singleStepForward}
            disabled={isForwardDisabled()}
          >
            <FaSolidForward /> Forward
          </button>
          <button
            class={`${styles.controlButton} ${isLossDisabled() ? styles.disabledButton : ''}`}
            onClick={calculateLoss}
            disabled={isLossDisabled()}
          >
            <FaSolidCalculator /> Loss
          </button>
        </div>
        <div class={styles.buttonGroup}>
          <button
            class={`${styles.controlButton} ${isBackwardDisabled() ? styles.disabledButton : ''}`}
            onClick={stepBackward}
            disabled={isBackwardDisabled()}
          >
            <FaSolidBackward /> Backward
          </button>
          <button
            class={`${styles.controlButton} ${isUpdateWeightsDisabled() ? styles.disabledButton : ''}`}
            onClick={updateWeights}
            disabled={isUpdateWeightsDisabled()}
          >
            <FaSolidWeightScale /> Update weights
          </button>
        </div>
      </div>
      <div class={styles.separator}></div>
      <button
        class={`${styles.resetButton} ${isResetDisabled() ? styles.disabledButton : ''}`}
        onClick={trainingStateReset}
        disabled={isResetDisabled()}
      >
        Reset
      </button>
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