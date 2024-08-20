import { Component, createEffect, createSignal, For, Show } from "solid-js";
import { css } from "@emotion/css";
import { actions, store } from '../store';
import LossHistoryChart from './LossHistoryChart';
import ForwardStepsVisualizer from './ForwardStepsVisualizer';

import TrainingControlButtons from "./TrainingControlButtons";
import TrainingStatus from "./TrainingStatus";


export const styles = {
  container: css`
    background-color: white;
    padding: 1.5rem;
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-top: 1rem;
  `,
  title: css`
    font-size: 1.5rem;
    font-weight: bold;
    margin-bottom: 1rem;
    color: #2c3e50;
  `,
  controlsContainer: css`
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  `,
  button: css`
    background-color: #3B82F6;
    color: white;
    border: none;
    border-radius: 0.25rem;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    cursor: pointer;
    transition: background-color 0.2s;
    &:hover {
      background-color: #2563EB;
    }
    &:disabled {
      background-color: #9CA3AF;
      cursor: not-allowed;
    }
  `,
  exportButton: css`
    background-color: #10B981;
    color: white;
    border: none;
    border-radius: 0.25rem;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    cursor: pointer;
    transition: background-color 0.2s;
    margin-top: 1rem;
    &:hover {
      background-color: #059669;
    }
  `,
  progressContainer: css`
    margin-bottom: 1rem;
  `,
  progressLabel: css`
    font-size: 0.875rem;
    color: #4B5563;
    margin-bottom: 0.25rem;
  `,
  progressBar: css`
    width: 100%;
    height: 0.5rem;
    background-color: #E5E7EB;
    border-radius: 0.25rem;
    overflow: hidden;
  `,
  progressFill: css`
    height: 100%;
    background-color: #3B82F6;
    transition: width 300ms ease-in-out;
  `,
  lossContainer: css`
    display: flex;
    align-items: center;
    margin-bottom: 1rem;
  `,
  lossLabel: css`
    font-size: 0.875rem;
    color: #4B5563;
    margin-right: 0.5rem;
  `,
  lossValue: css`
    font-size: 1rem;
    font-weight: bold;
  `,
};

const TrainingControls: Component<{
  onVisualizationUpdate: () => void
}> = (props) => {
  const [lossHistory, setLossHistory] = createSignal<number[]>([]);
  const [zoomRange, setZoomRange] = createSignal<[number, number]>([0, 100]);
  const [chartType, setChartType] = createSignal<'bar' | 'line'>('line');
  const [trainingRuns, setTrainingRuns] = createSignal<{ id: string; lossHistory: number[] }[]>([]);
  const [selectedRuns, setSelectedRuns] = createSignal<string[]>([]);
  const [isTraining, setIsTraining] = createSignal(false);

  createEffect(() => {
    if (store.trainingResult?.data.loss) {
      setLossHistory([...lossHistory(), store.trainingResult.data.loss]);
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
    if (loss < 0.2) return '#10B981';
    if (loss < 0.5) return '#FBBF24';
    return '#EF4444';
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
    props.onVisualizationUpdate();
  };

  const stepBackward = () => {
    actions.stepBackward();
    props.onVisualizationUpdate();
  };

  const updateWeights = () => {
    actions.updateWeights();
    props.onVisualizationUpdate();
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
      <TrainingControlButtons
        onSingleStepForward={singleStepForward}
        onStepBackward={stepBackward}
        onUpdateWeights={updateWeights}
      />
      <ForwardStepsVisualizer
        forwardStepsCount={store.forwardStepsCount}
        forwardStepResults={store.forwardStepResults}
        batchSize={store.trainingConfig.batchSize}
      />
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
      <button class={styles.exportButton} onClick={exportCSV}>Export CSV</button>
      {/* <RunSelector
        trainingRuns={trainingRuns()}
        selectedRuns={selectedRuns()}
        onToggleRunSelection={toggleRunSelection}
        onSaveCurrentRun={saveCurrentRun}
      />
      <LossStatistics
        minLoss={minLoss()}
        maxLoss={maxLoss()}
        avgLoss={avgLoss()}
      /> */}
    </div>
  );
};

export default TrainingControls;