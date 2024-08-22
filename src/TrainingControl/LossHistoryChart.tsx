import { Component, createEffect } from "solid-js";
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
  `,
  title: css`
    font-size: 1rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
    color: ${colors.text};
  `,
  chart: css`
    display: flex;
    align-items: flex-end;
    height: 100px;
    border-bottom: 1px solid ${colors.border};
    border-left: 1px solid ${colors.border};
  `,
  bar: css`
    width: 4px;
    background-color: ${colors.primary};
    margin-right: 2px;
    transition: height 0.3s ease;
  `,
};

const LossHistoryChart: Component = () => {
  const getLossHistory = () => store.trainingState.lossHistory || [];

  createEffect(() => {
    // This effect will run whenever the loss history changes
    getLossHistory();
  });

  return (
    <div class={styles.container}>
      <h4 class={styles.title}>Loss History</h4>
      <div class={styles.chart}>
        {getLossHistory().map((loss) => (
          <div
            class={styles.bar}
            style={{ height: `${Math.min(100, loss * 100)}%` }}
            title={`Loss: ${loss.toFixed(4)}`}
          />
        ))}
      </div>
    </div>
  );
};

export default LossHistoryChart;