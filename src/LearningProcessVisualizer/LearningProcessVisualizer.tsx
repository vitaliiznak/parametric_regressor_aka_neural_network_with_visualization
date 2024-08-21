import { Component, Show, For, createMemo } from "solid-js";
import { store } from "../store";
import { colors } from '../styles/colors';
import { css } from "@emotion/css";

const LearningProcessVisualizer: Component = () => {

  const renderData = createMemo(() => {
    if (!store.trainingResult) return null;

    const { step, data } = store.trainingResult;
    switch (step) {
      case 'forward':
        return (
          <div>
            <h4>Forward Pass</h4>
            <p>Input: {JSON.stringify(data.input)}</p>
            <p>Output: {JSON.stringify(data.output)}</p>
          </div>
        );
      case 'backward':
        return (
          <div>
            <h4>Backward Pass</h4>
            <p>Gradients: {data.gradients?.map(g => g.toFixed(4)).join(', ')}</p>
          </div>
        );
      case 'update':
        return (
          <div>
            <h4>Weight Update</h4>
            <For each={data.oldWeights}>
              {(oldWeight, index) => (
                <p>
                  Weight {index()}: {oldWeight.toFixed(4)} → {data.newWeights?.[index()].toFixed(4)}
                </p>
              )}
            </For>
            <p>Learning Rate: {data.learningRate}</p>
          </div>
        );
      case 'iteration':
        return <div>Iteration {data.iteration} completed, Loss: {data.loss?.toFixed(4)}</div>;
      default:
        return null;
    }
  });

  const styles = {
    container: css`
      background-color: ${colors.surface};
      padding: 1.5rem;
      border-radius: 0.5rem;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      margin-top: 1rem;
      
      @media (max-width: 768px) {
        padding: 1rem;
      }
    `,
    title: css`
      font-size: 1.25rem;
      font-weight: bold;
      margin-bottom: 1rem;
      color: ${colors.text};
    `,
    stepInfo: css`
      margin-bottom: 1rem;
      color: ${colors.textLight};
    `,
    dataDisplay: css`
      background-color: ${colors.background};
      padding: 1rem;
      border-radius: 0.25rem;
      font-family: monospace;
      white-space: pre-wrap;
      color: ${colors.text};
      font-size: 0.875rem;
      
      @media (max-width: 768px) {
        font-size: 0.75rem;
      }
    `,
  };

  return (
    <div class={styles.container}>
      <h3 class={styles.title}>Learning Process</h3>
      <Show when={store.trainingResult}>
        <div class={styles.stepInfo}>Current Step: {store.trainingResult?.step}</div>
        <div class={styles.dataDisplay}>{JSON.stringify(renderData(), null, 2)}</div>
      </Show>
    </div>
  );
};

export default LearningProcessVisualizer;