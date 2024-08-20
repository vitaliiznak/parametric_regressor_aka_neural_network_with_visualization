import { Component, createSignal, For } from "solid-js";
import { css } from "@emotion/css";

interface LossHistoryChartProps {
  lossHistory: number[];
  trainingRuns: { id: string; lossHistory: number[] }[];
  selectedRuns: string[];
  chartType: 'bar' | 'line';
  zoomRange: [number, number];
  maxLoss: number;
  handleWheel: (e: WheelEvent) => void;
  onChartTypeChange: (type: 'bar' | 'line') => void;
  onZoomRangeChange: (range: [number, number]) => void;
}

const LossHistoryChart: Component<LossHistoryChartProps> = (props) => {
  const [hoveredBar, setHoveredBar] = createSignal<number | null>(null);

  const styles = {
    container: css`
      margin-top: 1rem;
    `,
    chartTitle: css`
      font-size: 1.25rem;
      font-weight: bold;
      margin-bottom: 0.5rem;
    `,
    chart: css`
      width: 100%;
      height: 15rem;
      background-color: #f3f4f6;
      position: relative;
      border: 1px solid #d1d5db;
      overflow: hidden;
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
  };

  const renderChart = () => {
    const width = 100 / (props.lossHistory.length - 1);

    const renderRun = (run: { id: string; lossHistory: number[] }, color: string) => {
      if (props.chartType === 'bar') {
        return (
          <For each={run.lossHistory}>
            {(loss, index) => (
              <div
                style={{
                  position: 'absolute',
                  height: `${(loss / props.maxLoss) * 100}%`,
                  left: `${index() * width}%`,
                  width: `${width * 0.8}%`,
                  'background-color': color,
                  opacity: 0.7,
                  bottom: 0,
                }}
                onMouseEnter={() => setHoveredBar(index())}
                onMouseLeave={() => setHoveredBar(null)}
              >
                {hoveredBar() === index() && (
                  <div style={{
                    position: 'absolute',
                    bottom: '100%',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    'background-color': 'rgba(0, 0, 0, 0.8)',
                    color: 'white',
                    padding: '0.25rem 0.5rem',
                    'border-radius': '0.25rem',
                    'font-size': '0.75rem',
                  }}>
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
              points={run.lossHistory.map((loss, index) => `${index * width},${100 - (loss / props.maxLoss) * 100}`).join(' ')}
              fill="none"
              stroke={color}
              stroke-width="0.5"
            />
          </svg>
        );
      }
    };

    return (
      <div style={{ position: 'absolute', left: '2.5rem', right: '0', top: '0', bottom: '1.5rem' }}>
        {renderRun({ id: 'current', lossHistory: props.lossHistory }, '#3B82F6')}
        <For each={props.trainingRuns.filter(run => props.selectedRuns.includes(run.id))}>
          {(run, index) => renderRun(run, `hsl(${index() * 60}, 70%, 50%)`)}
        </For>
      </div>
    );
  };

  return (
    <div class={styles.container}>
      <h4 class={styles.chartTitle}>Loss History</h4>
      <select value={props.chartType} onChange={(e) => props.onChartTypeChange(e.target.value as 'bar' | 'line')}>
        <option value="bar">Bar Chart</option>
        <option value="line">Line Chart</option>
      </select>
      <div class={styles.chart} onWheel={props.handleWheel}>
        <div class={styles.yAxis}>
          <span>{props.maxLoss.toFixed(2)}</span>
          <span>{(props.maxLoss / 2).toFixed(2)}</span>
          <span>0.00</span>
        </div>
        {renderChart()}
        <div class={styles.xAxis}>
          <span>0</span>
          <span>{props.lossHistory.length - 1}</span>
        </div>
      </div>
      <div class={styles.zoomControl}>
        <span class={styles.zoomLabel}>Zoom:</span>
        <input
          type="range"
          min="0"
          max="100"
          step="1"
          value={props.zoomRange[0]}
          onInput={(e) => props.onZoomRangeChange([Number(e.target.value), props.zoomRange[1]])}
          class={styles.zoomInput}
        />
        <input
          type="range"
          min="0"
          max="100"
          step="1"
          value={props.zoomRange[1]}
          onInput={(e) => props.onZoomRangeChange([props.zoomRange[0], Number(e.target.value)])}
          class={styles.zoomInput}
        />
      </div>
    </div>
  );
};

export default LossHistoryChart;