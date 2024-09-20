import { Component, createEffect, createMemo, createSignal, Show } from "solid-js";
import { css, keyframes } from "@emotion/css";
import { actions, setStore, store } from '../store';
import TrainingStepsVisualizer from './TrainingStepsVisualizer';
import TrainingStatus from "./TrainingStatus";
import { colors } from '../styles/colors';
import { typography } from '../styles/typography';
import { commonStyles } from '../styles/common';
import { FaSolidBackward, FaSolidCalculator, FaSolidForward, FaSolidWeightScale } from "solid-icons/fa";
import LossHistoryChart from "./LossHistoryChart";

// Define keyframes for animations
const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(-20%); }
  to { opacity: 1; transform: translateY(0); }
`;

const fadeOut = keyframes`
  from { opacity: 1; transform: translateY(0); }
  to { opacity: 0; transform: translateY(-20%); }
`;

const styles = {
  container: css`
    ${commonStyles.card}
    margin-top: 1rem;
    position: relative;
  `,
  title: css`
    font-size: ${typography.fontSize.xl};
    font-weight: ${typography.fontWeight.bold};
    margin-bottom: 0.5rem;
    color: ${colors.text};
  `,
  controlsContainer: css`
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.25rem;
    margin-bottom: 0.5rem;

    @media (max-width: 600px) {
      grid-template-columns: repeat(2, 1fr);
    }
  `,
  controlButton: css`
    ${commonStyles.button}
    ${commonStyles.primaryButton}
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0.25rem;
    border-radius: 4px;
    font-size: ${typography.fontSize.base};

    &:hover:not(:disabled) {
      transform: none;
      box-shadow: none;
    }

    span {
      margin-left: 4px;
      font-size: ${typography.fontSize.xxxs};
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
    padding: 0.5rem 1rem;
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
    padding: 0.25rem;
    border-radius: 4px;
    font-size: ${typography.fontSize.xxs};

    &:hover:not(:disabled) {
      transform: none;
      box-shadow: none;
    }
  `,
  iterationIndicator: css`
    text-align: center;
    margin: 1rem 0;
    font-size: ${typography.fontSize.xl};
    font-weight: ${typography.fontWeight.bold};
    color: ${colors.primary};
    animation: ${fadeIn} 0.5s ease-in-out;
  `,
  notification: css`
    position: fixed;
    top: 20%;
    left: 50%;
    transform: translate(-50%, -50%);
    background-color: ${colors.primary};
    color: ${colors.textLight};
    padding: 1rem 2rem;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    animation: ${fadeIn} 0.5s ease-in-out, ${fadeOut} 0.5s ease-in-out 2.5s;
    z-index: 1000;
  `,
};

const TrainingControls: Component = () => {
  const [zoomRange, setZoomRange] = createSignal<[number, number]>([0, 100]);
  const [chartType, setChartType] = createSignal<'bar' | 'line'>('line');
  const [showNotification, setShowNotification] = createSignal(false);
  const [currentIteration, setCurrentIteration] = createSignal(1);

  createEffect(() => {
    if (
      store.trainingState.currentPhase === 'update' &&
      store.trainingState.backwardStepGradients.length === 0
    ) {
      setShowNotification(true);
      setTimeout(() => setShowNotification(false), 3000);
      // Increment iteration count
      setCurrentIteration(prev => prev + 1);
      // Reset for the next iteration
      actions.trainingStateReset();
      // Set the current phase to 'idle' to enable the Forward button

    }
  });

  const iterationProgress = () => {
    const iteration = store.trainingState.iteration || 0;
    const totalIterations = store.trainingConfig?.iterations || 1;
    return iteration / totalIterations;
  };

  const getLossColor = (loss: number | null) => {
    if (loss === null) return colors.error;
    if (loss < 0.2) return colors.success;
    if (loss < 0.5) return colors.warning;
    return colors.error;
  };

  const isForwardDisabled = createMemo(() => { 
    console.log('isForwardDisabled store.trainingState.currentPhase', store.trainingState.currentPhase)
    return store.trainingState.currentPhase !== 'idle' && store.trainingState.currentPhase !== 'forward'
  })

  const isLossDisabled = createMemo(() =>
    store.trainingState.forwardStepResults.length === 0 ||
    store.trainingState.currentPhase !== 'forward')

  const isBackwardDisabled = createMemo(() => store.trainingState.currentPhase !== 'loss')
  const isUpdateWeightsDisabled = createMemo(() => store.trainingState.currentPhase !== 'backward')
  const isResetDisabled = createMemo(() => store.trainingState.forwardStepResults.length === 0)

  const singleStepForward = () => {
    actions.singleStepForward();
  };

  const trainingStateReset = () => {
    actions.trainingStateReset();
  };

  const calculateLoss = () => {
    actions.calculateLoss();
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
      <Show when={showNotification()}>
        <div class={styles.notification}>
          Iteration {currentIteration() - 1} Completed! Starting Iteration {currentIteration()}...
        </div>
      </Show>
      <div class={styles.iterationIndicator}>
        Current Iteration: {currentIteration()}
      </div>
      <TrainingStatus
        iteration={store.trainingState.iteration || 0}
        totalIterations={store.trainingConfig?.iterations || 1}
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
        <button
          class={styles.controlButton}
          onClick={singleStepForward}
          disabled={isForwardDisabled()}
          aria-label="Forward"
        >
          <FaSolidForward />
          <span>Forward</span>
        </button>
        <button
          class={styles.controlButton}
          onClick={calculateLoss}
          disabled={isLossDisabled()}
          aria-label="Calculate Loss"
        >
          <FaSolidCalculator />
          <span>Loss</span>
        </button>
        <button
          class={styles.controlButton}
          onClick={stepBackward}
          disabled={isBackwardDisabled()}
          aria-label="Backward"
        >
          <FaSolidBackward />
          <span>Backward</span>
        </button>
        <button
          class={styles.controlButton}
          onClick={updateWeights}
          disabled={isUpdateWeightsDisabled()}
          aria-label="Update Weights"
        >
          <FaSolidWeightScale />
          <span>Update</span>
        </button>
      </div>
      <div class={styles.separator}></div>
      <button
        class={styles.resetButton}
        onClick={trainingStateReset}
        disabled={isResetDisabled()}
        aria-label="Reset"
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