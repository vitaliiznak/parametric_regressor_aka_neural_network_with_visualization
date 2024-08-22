import { Component, createEffect, createSignal, For, Show } from "solid-js";
import { css } from "@emotion/css";
import { actions, setStore, store } from '../store';
import TrainingStepsVisualizer from './TrainingStepsVisualizer';

import TrainingStatus from "./TrainingStatus";
import { colors } from '../styles/colors';
import { FaSolidBackward, FaSolidCalculator, FaSolidForward } from "solid-icons/fa";
import LossHistoryChart from "./LossHistoryChart";

export const styles = {
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
  &:disabled {
    background-color: #E5E7EB;
    cursor: not-allowed;
  }
`,
  container: css`
    background-color: ${colors.surface};
    padding: 1.5rem;
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-top: 1rem;
  `,
  title: css`
    font-size: 1.25rem;
    font-weight: bold;
    margin-bottom: 1rem;
    color: ${colors.text};
  `,
  button: css`
    background-color: ${colors.primary};
    color: ${colors.surface};
    border: none;
    border-radius: 0.25rem;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    cursor: pointer;
    transition: background-color 0.2s;
    &:hover {
      background-color: ${colors.primaryDark};
    }
    &:disabled {
      background-color: ${colors.border};
      cursor: not-allowed;
    }
  `,
  exportButton: css`
    background-color: ${colors.secondary};
    color: ${colors.surface};
    border: none;
    border-radius: 0.25rem;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    cursor: pointer;
    transition: background-color 0.2s;
    margin-top: 1rem;
    &:hover {
      background-color: ${colors.secondaryDark};
    }
  `,
  progressContainer: css`
    margin-bottom: 1rem;
  `,
  progressLabel: css`
    font-size: 0.875rem;
    color: ${colors.textLight};
    margin-bottom: 0.25rem;
  `,
  progressBar: css`
    width: 100%;
    height: 0.5rem;
    background-color: ${colors.border};
    border-radius: 0.25rem;
    overflow: hidden;
  `,
  progressFill: css`
    height: 100%;
    background-color: ${colors.primary};
    transition: width 300ms ease-in-out;
  `,
  lossContainer: css`
    display: flex;
    align-items: center;
    margin-bottom: 1rem;
  `,
  lossLabel: css`
    font-size: 0.875rem;
    color: ${colors.textLight};
    margin-right: 0.5rem;
  `,
  lossValue: css`
    font-size: 1rem;
    font-weight: bold;
  `,
};

const TrainingControls: Component = () => {
  const [zoomRange, setZoomRange] = createSignal<[number, number]>([0, 100]);
  const [chartType, setChartType] = createSignal<'bar' | 'line'>('line');
  const [isLossCalculated, setIsLossCalculated] = createSignal(false);

  createEffect(() => {
    const {currentPhase} = store.trainingState 
    if (currentPhase === 'forward') {
      setStore('trainingState', 'forwardStepsCount', store.trainingState.forwardStepsCount + 1);
      setIsLossCalculated(false);
    } else if (currentPhase === 'loss') {
      setIsLossCalculated(true);
    } else if (currentPhase === 'backward') {
      setStore('trainingState', 'forwardStepsCount', 0);
      setIsLossCalculated(false);
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
    if (store.trainingState.forwardStepsCount === 0) {
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
      <h3 class={styles.title}>Training Control</h3>
      <TrainingStatus
        iteration={store.trainingState.iteration || 0}
        totalIterations={store.trainingConfig?.iterations || 0}
        currentLoss={store.trainingState.currentLoss}
        iterationProgress={iterationProgress()}
        getLossColor={getLossColor}
      />

      <TrainingStepsVisualizer
        forwardStepResults={store.trainingState.forwardStepResults}
        batchSize={store.trainingState.forwardStepResults.length}
        currentLoss={store.trainingState.currentLoss}
      />

      <div class={styles.controlsContainer}>
        <button class={styles.controlButton} onClick={singleStepForward}>
          <FaSolidForward /> Forward Step
        </button>
      

      
        <Show when={store.trainingState.forwardStepsCount > 0}>
          <button class={styles.button} onClick={calculateLoss}>
            <FaSolidCalculator /> Calculate Loss
          </button>
        </Show>
        <Show when={isLossCalculated()}>
          <button class={styles.controlButton} onClick={stepBackward}>
            <FaSolidBackward /> Backward Step
          </button>
        </Show>
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