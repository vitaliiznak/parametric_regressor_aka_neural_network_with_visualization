import { Component, createEffect, createSignal, For, Show } from "solid-js";
import { css, keyframes } from "@emotion/css";
import { FaSolidPlay, FaSolidPause, FaSolidForward, FaSolidBackward, FaSolidStop } from 'solid-icons/fa';
import { actions, store } from '../store';



const TrainingControls: Component<{
  onVisualizationUpdate: () => void
}> = (
  onVisualizationUpdate
) => {
  const [lossHistory, setLossHistory] = createSignal<number[]>([]);
  const [hoveredBar, setHoveredBar] = createSignal<number | null>(null);
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

  const epochProgress = () => {
    const currentEpoch = store.trainingResult?.data.epoch || 0;
    const totalEpochs = store.trainingConfig?.epochs || 1;
    return currentEpoch / totalEpochs;
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

  const pulse = keyframes`
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
  `;

  const styles = {
    container: css`
      background-color: white;
      padding: 1.5rem;
      border-radius: 0.5rem;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      margin-top: 1rem;
    `,
    title: css`
      font-size: 1.25rem;
      font-weight: bold;
      margin-bottom: 1rem;
    `,
    chartContainer: css`
      margin-top: 1rem;
    `,
    chartTitle: css`
      font-weight: bold;
      margin-bottom: 0.5rem;
    `,
    chart: css`
      width: 100%;
      height: 15rem;
      background-color: #f3f4f6;
      position: relative;
      border: 1px solid #d1d5db;
    `,
    yAxis: css`
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      width: 2.5rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      font-size: 0.75rem;
      color: #4b5563;
      padding: 0.25rem 0;
    `,
    xAxis: css`
      position: absolute;
      left: 2.5rem;
      right: 0;
      bottom: 0;
      height: 1.5rem;
      display: flex;
      justify-content: space-between;
      font-size: 0.75rem;
      color: #4b5563;
      padding-top: 0.25rem;
    `,
    bar: css`
      position: absolute;
      bottom: 0;
      width: 0.25rem;
      transition: all 300ms ease-in-out;
      cursor: pointer;
      &:hover {
        width: 0.5rem;
      }
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
    statusGrid: css`
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 1rem;
      margin-bottom: 1rem;
    `,
    statusItem: css`
      background-color: #F3F4F6;
      padding: 1rem;
      border-radius: 0.5rem;
      transition: all 0.3s ease;
      &:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
      }
    `,
    statusLabel: css`
      font-size: 0.875rem;
      color: #4B5563;
      margin-bottom: 0.25rem;
    `,
    statusValue: css`
      font-size: 1.25rem;
      font-weight: bold;
      animation: ${pulse} 2s infinite;
    `,
    tooltip: css`
      position: absolute;
      background-color: rgba(0, 0, 0, 0.8);
      color: white;
      padding: 0.25rem 0.5rem;
      border-radius: 0.25rem;
      font-size: 0.75rem;
      pointer-events: none;
      transition: all 0.2s ease;
    `,
    zoomControl: css`
      margin-top: 1rem;
      display: flex;
      align-items: center;
    `,
    zoomLabel: css`
      margin-right: 0.5rem;
      font-size: 0.875rem;
    `,
    zoomInput: css`
      width: 100%;
    `,
    statsGrid: css`
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 1rem;
      margin-top: 1rem;
    `,
    statItem: css`
      background-color: #F3F4F6;
      padding: 0.5rem;
      border-radius: 0.25rem;
      text-align: center;
    `,
    statLabel: css`
      font-size: 0.75rem;
      color: #4B5563;
    `,
    statValue: css`
      font-size: 1rem;
      font-weight: bold;
    `,
    exportButton: css`
      margin-top: 1rem;
    `,
    chartTypeSelect: css`
      margin-top: 1rem;
    `,
    runSelector: css`
      margin-top: 1rem;
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    `,
    runCheckbox: css`
      display: flex;
      align-items: center;
      background-color: #F3F4F6;
      padding: 0.25rem 0.5rem;
      border-radius: 0.25rem;
      font-size: 0.875rem;
      cursor: pointer;
      transition: background-color 0.2s;
      &:hover {
        background-color: #E5E7EB;
      }
    `,
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
      &:hover {
        background-color: #2563EB;
      }
    `,
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
    if (!store.trainingWorker) {
      actions.startTraining();
    } else if (isTraining()) {
      actions.pauseTraining();
    } else {
      actions.resumeTraining();
    }
    setIsTraining(!isTraining());
  };

  const startTraining = () => {
    if (!store.trainingWorker) {
      actions.startTraining();
      setIsTraining(true);
    }
  };

  const stepForward = () => {
    // TODO: Implement step forward logic
  };

  const stepBackward = () => {
    // TODO: Implement step backward logic
  };

  const renderChart = () => {
    const chartData = visibleLossHistory();
    const maxY = maxLoss();
    const width = 100 / (chartData.length - 1);

    const renderRun = (run: { id: string; lossHistory: number[] }, color: string) => {
      if (chartType() === 'bar') {
        return (
          <For each={run.lossHistory}>
            {(loss, index) => (
              <div
                class={styles.bar}
                style={{
                  height: `${(loss / maxY) * 100}%`,
                  left: `${index() * width}%`,
                  width: `${width * 0.8}%`,
                  backgroundColor: color,
                  opacity: 0.7,
                }}
                onMouseEnter={() => setHoveredBar(index())}
                onMouseLeave={() => setHoveredBar(null)}
              >
                {hoveredBar() === index() && (
                  <div class={styles.tooltip} style={{ bottom: '100%', left: '50%', transform: 'translateX(-50%)' }}>
                    Loss: {loss.toFixed(4)}
                  </div>
                )}
              </div>
            )}
          </For>
        );
      } else {
        return (
          <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none">
            <polyline
              points={run.lossHistory.map((loss, index) => `${index * width},${100 - (loss / maxY) * 100}`).join(' ')}
              fill="none"
              stroke={color}
              stroke-width="0.5"
            />
            <For each={run.lossHistory}>
              {(loss, index) => (
                <circle
                  cx={`${index() * width}%`}
                  cy={`${100 - (loss / maxY) * 100}%`}
                  r="0.5"
                  fill={color}
                  onMouseEnter={() => setHoveredBar(index())}
                  onMouseLeave={() => setHoveredBar(null)}
                >
                  {hoveredBar() === index() && (
                    <title>Loss: {loss.toFixed(4)}</title>
                  )}
                </circle>
              )}
            </For>
          </svg>
        );
      }
    };

    return (
      <div style={{ position: 'absolute', left: '2.5rem', right: '0', top: '0', bottom: '1.5rem' }}>
        {renderRun({ id: 'current', lossHistory: chartData }, '#3B82F6')}
        <For each={trainingRuns().filter(run => selectedRuns().includes(run.id))}>
          {(run, index) => renderRun(run, `hsl(${index() * 60}, 70%, 50%)`)}
        </For>
      </div>
    );
  };

  return (
    <div class={styles.container}>
      <h3 class={styles.title}>Training Control</h3>
      <div class={styles.statusGrid}>
        <div class={styles.statusItem}>
          <div class={styles.statusLabel}>Epoch</div>
          <div class={styles.statusValue}>
            {store.trainingResult?.data.epoch || 0} / {store.trainingConfig?.epochs || 0}
          </div>
          <div class={styles.progressBar}>
            <div class={styles.progressFill} style={{ width: `${epochProgress() * 100}%` }}></div>
          </div>
        </div>
        <div class={styles.statusItem}>
          <div class={styles.statusLabel}>Current Loss</div>
          <div class={styles.statusValue} style={{ color: getLossColor(store.trainingResult?.data.loss || 0) }}>
            {store.trainingResult?.data.loss?.toFixed(4) || 'N/A'}
          </div>
        </div>
      </div>
      <div class={styles.controlsContainer}>
        <Show when={!store.trainingWorker}>
          <button class={styles.controlButton} onClick={startTraining}>Start Training</button>
        </Show>
        <Show when={store.trainingWorker}>
          <button class={styles.controlButton} onClick={toggleTraining}>
            {isTraining() ? <FaSolidPause /> : <FaSolidPlay />}
          </button>
          <button class={styles.controlButton} onClick={actions.stopTraining}><FaSolidStop /></button>
        </Show>
      </div>
      <div class={styles.chartContainer}>
        <h4 class={styles.chartTitle}>Loss History</h4>
        <div class={styles.chartTypeSelect}>
          <select value={chartType()} onChange={(e) => setChartType(e.target.value as 'bar' | 'line')}>
            <option value="bar">Bar Chart</option>
            <option value="line">Line Chart</option>
          </select>
        </div>
        <div class={styles.chart}>
          <div class={styles.yAxis}>
            <span>{maxLoss().toFixed(2)}</span>
            <span>{(maxLoss() / 2).toFixed(2)}</span>
            <span>0.00</span>
          </div>
          {renderChart()}
          <div class={styles.xAxis}>
            <span>0</span>
            <span>{lossHistory().length - 1}</span>
          </div>
        </div>
        <div class={styles.zoomControl}>
          <span class={styles.zoomLabel}>Zoom:</span>
          <input
            type="range"
            min="0"
            max="100"
            step="1"
            value={zoomRange()[0]}
            onInput={(e) => setZoomRange([Number(e.target.value), zoomRange()[1]])}
            class={styles.zoomInput}
          />
          <input
            type="range"
            min="0"
            max="100"
            step="1"
            value={zoomRange()[1]}
            onInput={(e) => setZoomRange([zoomRange()[0], Number(e.target.value)])}
            class={styles.zoomInput}
          />
        </div>
        <button class={styles.exportButton} onClick={exportCSV}>Export CSV</button>
        <div class={styles.runSelector}>
          <For each={trainingRuns()}>
            {(run) => (
              <label class={styles.runCheckbox}>
                <input
                  type="checkbox"
                  checked={selectedRuns().includes(run.id)}
                  onChange={() => toggleRunSelection(run.id)}
                />
                Run {run.id}
              </label>
            )}
          </For>
        </div>
        <button onClick={saveCurrentRun}>Save Current Run</button>
      </div>
      <Show when={lossHistory().length > 0}>
        <div class={styles.statsGrid}>
          <div class={styles.statItem}>
            <div class={styles.statLabel}>Min Loss</div>
            <div class={styles.statValue}>{minLoss().toFixed(4)}</div>
          </div>
          <div class={styles.statItem}>
            <div class={styles.statLabel}>Max Loss</div>
            <div class={styles.statValue}>{maxLoss().toFixed(4)}</div>
          </div>
          <div class={styles.statItem}>
            <div class={styles.statLabel}>Avg Loss</div>
            <div class={styles.statValue}>{avgLoss().toFixed(4)}</div>
          </div>
        </div>
      </Show>
    </div>
  );
};

export default TrainingControls;