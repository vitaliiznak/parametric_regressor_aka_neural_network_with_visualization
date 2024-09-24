import { Component, createMemo, createSignal, For, Show } from "solid-js";
import { store } from "../store";
import { css } from "@emotion/css";
import { colors } from "../styles/colors";

const styles = {
  container: css`
    margin-top: 1rem;
    padding: 1rem;
    background-color: ${colors.surface};
    border-radius: 0.5rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    width: 100%;
    box-sizing: border-box;
  `,
  title: css`
    font-size: 1.2rem;
    font-weight: bold;
    margin-bottom: 0.75rem;
    color: ${colors.text};
    text-align: center;
  `,
  chartContainer: css`
    display: flex;
    align-items: flex-end;
    height: 200px; /* Increased height for better label accommodation */
    position: relative;
  `,
  yAxis: css`
    width: 60px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    margin-right: 10px;
    color: ${colors.textLight};
    font-size: 0.8rem;
    box-sizing: border-box;
  `,
  yAxisLabel: css`
    text-align: right;
    padding-right: 5px;
  `,
  chart: css`
    display: flex;
    align-items: flex-end;
    height: 100%;
    border-left: 2px solid ${colors.border};
    border-bottom: 2px solid ${colors.border};
    position: relative;
    flex-grow: 1;
    overflow-x: auto;
    box-sizing: border-box;
  `,
  bar: css`
    width: 5px;
    background-color: ${colors.primary};
    margin-right: 3px;
    transition: height 0.3s ease;
    position: relative;
    cursor: pointer;
    
    &:hover {
      background-color: ${colors.primaryDark};
    }
  `,
  tooltip: css`
    position: absolute;
    background-color: ${colors.surface};
    border: 1px solid ${colors.border};
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    pointer-events: none;
    transform: translate(-50%, -120%);
    white-space: nowrap;
    z-index: 10;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  `,
};

interface TooltipState {
  visible: boolean;
  x: number;
  y: number;
  content: string;
}

function niceNumber(range: number, round: boolean): number {
  const exponent = Math.floor(Math.log10(range));
  const fraction = range / Math.pow(10, exponent);
  let niceFraction: number;

  if (round) {
    if (fraction < 1.5) {
      niceFraction = 1;
    } else if (fraction < 3) {
      niceFraction = 2;
    } else if (fraction < 7) {
      niceFraction = 5;
    } else {
      niceFraction = 10;
    }
  } else {
    if (fraction <= 1) {
      niceFraction = 1;
    } else if (fraction <= 2) {
      niceFraction = 2;
    } else if (fraction <= 5) {
      niceFraction = 5;
    } else {
      niceFraction = 10;
    }
  }

  return niceFraction * Math.pow(10, exponent);
}

const LossHistoryChart: Component = () => {
  const getLossHistory = () => store.trainingState.lossHistory || [];
  const currentLoss = () => store.trainingState.currentLoss;

  // Signal to manage tooltip state
  const [tooltip, setTooltip] = createSignal<TooltipState>({
    visible: false,
    x: 0,
    y: 0,
    content: "",
  });

  // Memo to compute dynamic Y-axis scale based on loss history and current loss
  const yScale = createMemo(() => {
    const history = getLossHistory();
    const latestLoss = currentLoss();
  
    // Combine history and current loss to ensure current loss is included in scaling
    const combinedLosses = latestLoss !== null ? [...history, latestLoss] : history;
  
    if (combinedLosses.length === 0) {
      return { adjustedMin: 0, adjustedMax: 1, labels: [0, 0.2, 0.4, 0.6, 0.8, 1] };
    }
  
    const maxLoss = Math.max(...combinedLosses);
    const minLoss = Math.min(...combinedLosses);
  
    // Calculate range and nice range
    const range = niceNumber(maxLoss - minLoss, false);
    const niceMin = Math.floor(minLoss / Math.pow(10, Math.floor(Math.log10(range)))) * Math.pow(10, Math.floor(Math.log10(range)));
    const niceMax = Math.ceil(maxLoss / Math.pow(10, Math.floor(Math.log10(range)))) * Math.pow(10, Math.floor(Math.log10(range)));
  
    // Determine number of intervals based on range
    const numberOfIntervals = 5;
    const interval = (niceMax - niceMin) / numberOfIntervals;
    
    const labels = Array.from({ length: numberOfIntervals + 1 }, (_, i) =>
      parseFloat((niceMin + interval * i).toFixed(2))
    );
  
    return { adjustedMin: niceMin, adjustedMax: niceMax, labels };
  });

  // Handler for displaying tooltip on mouse move
  const handleMouseMove = (event: MouseEvent, loss: number) => {
    const chartElement = event.currentTarget as HTMLElement;
    const rect = chartElement.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    setTooltip({
      visible: true,
      x,
      y,
      content: `Loss: ${loss.toFixed(4)}`,
    });
  };

  // Handler to hide tooltip on mouse out
  const handleMouseOut = () => {
    setTooltip({
      visible: false,
      x: 0,
      y: 0,
      content: "",
    });
  };

  return (
    <div class={styles.container}>
      <h4 class={styles.title}>Loss History</h4>
      <div class={styles.chartContainer}>
        {/* Y-Axis Labels */}
        <div class={styles.yAxis}>
          <For each={yScale().labels}>
            {(label) => (
              <div class={styles.yAxisLabel}>
                {label}
              </div>
            )}
          </For>
        </div>
        {/* Bars */}
        <div class={styles.chart}>
          <For each={getLossHistory()}>
            {(loss, index) => {
              const heightPercentage =
              ((loss - yScale().adjustedMin) / (yScale().adjustedMax - yScale().adjustedMin)) * 100;

              return (
                <div
                  class={styles.bar}
                  style={{ height: `${heightPercentage}%` }}
                  onMouseMove={(e) => handleMouseMove(e, loss)}
                  onMouseOut={handleMouseOut}
                  role="button"
                  aria-label={`Loss at step ${index() + 1}: ${loss.toFixed(4)}`}
                >
                  <Show when={tooltip().visible && tooltip().content === `Loss: ${loss.toFixed(4)}`}>
                    <div
                      class={styles.tooltip}
                      style={{
                        left: "50%", // Center the tooltip horizontally over the bar
                        top: `${100 - heightPercentage}%`, // Position the tooltip above the bar
                      }}
                    >
                      {tooltip().content}
                    </div>
                  </Show>
                </div>
              );
            }}
          </For>
          {/* Display Current Loss as the last bar if it exists and not already in history */}
          <Show when={currentLoss() !== null}>
            <div
              class={styles.bar}
              style={{
                height: `${((currentLoss()! - yScale().adjustedMin) / (yScale().adjustedMax - yScale().adjustedMin)) * 100}%`,
                backgroundColor: colors.secondary,
              }}
              onMouseMove={(e) => handleMouseMove(e, currentLoss()!)}
              onMouseOut={handleMouseOut}
              role="button"
              aria-label={`Current Loss: ${currentLoss()!.toFixed(4)}`}
            >
              <Show when={tooltip().visible && tooltip().content === `Loss: ${currentLoss()!.toFixed(4)}`}>
                <div
                  class={styles.tooltip}
                  style={{
                    left: "50%",
                    top: `${100 - ((currentLoss()! - yScale().adjustedMin) / (yScale().adjustedMax - yScale().adjustedMin)) * 100}%`,
                  }}
                >
                  {`Current Loss: ${currentLoss()!.toFixed(4)}`}
                </div>
              </Show>
            </div>
          </Show>
        </div>
      </div>
    </div>
  );
};

export default LossHistoryChart;