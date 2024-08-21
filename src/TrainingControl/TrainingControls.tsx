import { Component, createEffect, createSignal, For, Show } from "solid-js";
import { css } from "@emotion/css";
import { actions, store } from '../store';
import LossHistoryChart from './LossHistoryChart';
import ForwardStepsVisualizer from './ForwardStepsVisualizer';

import TrainingStatus from "./TrainingStatus";
import { colors } from '../styles/colors';
import Tooltip from '../components/Tooltip';
import { FaSolidBackward, FaSolidCalculator, FaSolidForward } from "solid-icons/fa";

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
  const [lossHistory, setLossHistory] = createSignal<number[]>([]);
  const [zoomRange, setZoomRange] = createSignal<[number, number]>([0, 100]);
  const [chartType, setChartType] = createSignal<'bar' | 'line'>('line');
  const [trainingRuns, setTrainingRuns] = createSignal<{ id: string; lossHistory: number[] }[]>([]);
  const [selectedRuns, setSelectedRuns] = createSignal<string[]>([]);
  const [isTraining, setIsTraining] = createSignal(false);
  const [forwardStepsCount, setForwardStepsCount] = createSignal(0);
  const [isLossCalculated, setIsLossCalculated] = createSignal(false);

  createEffect(() => {
    if (store.trainingResult?.data.loss) {
      setLossHistory([...lossHistory(), store.trainingResult.data.loss]);
    }
  });

  createEffect(() => {
    if (store.trainingResult?.step === 'forward') {
      setForwardStepsCount(prev => prev + 1);
      setIsLossCalculated(false);
    } else if (store.trainingResult?.step === 'loss') {
      setIsLossCalculated(true);
    } else if (store.trainingResult?.step === 'backward') {
      setForwardStepsCount(0);
      setIsLossCalculated(false);
    }
  });

  const maxLoss = () => Math.max(...trainingRuns().flatMap(run => run.lossHistory), ...lossHistory(), 1);
  const minLoss = () => Math.min(...trainingRuns().flatMap(run => run.lossHistory), ...lossHistory(), 0);
  const avgLoss = () => {
    const allLosses = [...trainingRuns().flatMap(run => run.lossHistory), ...lossHistory()];
    return allLosses.reduce((sum, loss) => sum + loss, 0) / allLosses.length;
  };

  const iterationProgress = () => {
    const currentIteration = store.trainingResult?.data.iteration || 0;
    const totalIterations = store.trainingConfig?.iterations || 1;
    return currentIteration / totalIterations;
  };

  const getLossColor = (loss: number) => {
    if (loss < 0.2) return colors.success;
    if (loss < 0.5) return colors.error;
    return colors.error;
  };

  const visibleLossHistory = () => {
    const [start, end] = zoomRange();
    const startIndex = Math.floor(lossHistory().length * start / 100);
    const endIndex = Math.ceil(lossHistory().length * end / 100);
    return lossHistory().slice(startIndex, endIndex);
  };

  const exportCSV = () => {
    const csvContent = "data:text/csv;charset=utf-8,"
      + lossHistory().map((loss, index) => `${index},${loss}`).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "loss_history.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const saveCurrentRun = () => {
    const newRun = { id: Date.now().toString(), lossHistory: [...lossHistory()] };
    setTrainingRuns([...trainingRuns(), newRun]);
    setSelectedRuns([...selectedRuns(), newRun.id]);
  };

  const toggleRunSelection = (id: string) => {
    if (selectedRuns().includes(id)) {
      setSelectedRuns(selectedRuns().filter(runId => runId !== id));
    } else {
      setSelectedRuns([...selectedRuns(), id]);
    }
  };

  const toggleTraining = () => {
    if (!store.trainer) {
      actions.startTraining();
    } else if (isTraining()) {
      actions.pauseTraining();
    } else {
      actions.resumeTraining();
    }
    setIsTraining(!isTraining());
  };

  const singleStepForward = () => {
    actions.singleStepForward();
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
      <h3 class={styles.title}>Training Control</h3>
      <TrainingStatus
        iteration={store.trainingResult?.data.iteration || 0}
        totalIterations={store.trainingConfig?.iterations || 0}
        currentLoss={store.trainingResult?.data.loss || 0}
        iterationProgress={iterationProgress()}
        getLossColor={getLossColor}
      />



      <ForwardStepsVisualizer
        forwardStepsCount={store.forwardStepsCount}
        forwardStepResults={store.forwardStepResults}
        batchSize={store.trainingConfig.batchSize}
      />

      <div class={styles.controlsContainer}>
        <button class={styles.controlButton} onClick={singleStepForward}>
          <FaSolidForward /> Forward Step
        </button>
        <Show when={forwardStepsCount() >= 2}>
          <button
            class={styles.controlButton}
            onClick={calculateLoss}
            disabled={isLossCalculated()}
          >
            <FaSolidCalculator /> Calculate Loss
          </button>
        </Show>

        <button class={styles.button} onClick={calculateLoss} disabled={store.forwardStepsCount < store.trainingConfig.batchSize}>
          Calculate Loss
        </button>
        <Show when={isLossCalculated()}>
          <button class={styles.controlButton} onClick={stepBackward}>
            <FaSolidBackward /> Backward Step
          </button>
        </Show>
      </div>

      {/* <Tooltip content="View the loss history over training iterations">
        <LossHistoryChart
          lossHistory={visibleLossHistory()}
          trainingRuns={trainingRuns()}
          selectedRuns={selectedRuns()}
          chartType={chartType()}
          zoomRange={zoomRange()}
          maxLoss={maxLoss()}
          handleWheel={handleWheel}
          onChartTypeChange={(type) => setChartType(type)}
          onZoomRangeChange={setZoomRange}
        />
      </Tooltip>
      <Tooltip content="Export the loss history data as a CSV file">
        <button class={styles.exportButton} onClick={exportCSV}>Export CSV</button>
    */}
    </div>
  );
};

export default TrainingControls;