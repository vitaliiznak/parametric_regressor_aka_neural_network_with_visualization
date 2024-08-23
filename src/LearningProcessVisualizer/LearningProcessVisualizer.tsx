import { Component, Show, For, createMemo } from "solid-js";
import { store } from "../store";
import { colors } from '../styles/colors';
import { css } from "@emotion/css";
import { FaSolidForward, FaSolidBackward, FaSolidCalculator } from 'solid-icons/fa';

const LearningProcessVisualizer: Component = () => {
  console.log("LearningProcessVisualizer rendering");

  const renderData = createMemo(() => {
    console.log("renderData memo running");
    if (!store.trainingStepResult) {
      console.log("No training result");
      return null;
    }

    const { currentPhase } = store.trainingState;
    console.log("Rendering step:", currentPhase, "with data:");

    try {
      switch (currentPhase) {
        case 'forward':
          console.log("Rendering forward step");
          return (
            <div>
              <h4>Forward Pass</h4>
              {/* <p>Input: {JSON.stringify(data.input)}</p>
              <p>Output: {JSON.stringify(data.output)}</p> */}
            </div>
          );
        case 'loss':
          console.log("Rendering loss step");
          return (
            <div>
              <h4>Loss Calculation</h4>
              {/* <p>Loss: {data.loss?.toFixed(4)}</p> */}
            </div>
          );
        case 'backward':
          console.log("Rendering backward step");
          return (
            <div>
              <h4>Backward Pass</h4>
              {/* <p>Gradients: {data.gradients?.map(g => g.toFixed(4)).join(', ')}</p> */}
            </div>
          );
        case 'update':
          console.log("Rendering update step");
          return (
            <div>
              <h4>Weight Update</h4>
              {/* <For each={data.oldWeights}>
                {(oldWeight, index) => (
                  <p>
                    Weight {index()}: {oldWeight.toFixed(4)} → {data.newWeights?.[index()].toFixed(4)}
                  </p>
                )}
              </For>
              <p>Learning Rate: {data.learningRate}</p> */}
            </div>
          );
        case 'iteration':
          console.log("Rendering iteration step");
          return {/* <div>Iteration {data.iteration} completed, Loss: {data.loss?.toFixed(4)}</div>; */}
        default:
          console.log("Unknown step:", currentPhase);
          return null;
      }
    } catch (error) {
      console.error("Error in renderData:", error);
      return <div>Error rendering data</div>;
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
      <Show when={store.trainingStepResult}>
        {() => {
          console.log("Rendering training result");
          return (
            <>
              <div class={styles.stepInfo}>Current Step: {store.trainingState.currentPhase}</div>
              <Show when={store.trainingStepResult}>
                <div class={styles.dataDisplay}>
                  {JSON.stringify(renderData(), null, 2)}
                </div>
              </Show>
            </>
          );
        }}
      </Show>
    </div>
  );
};

export default LearningProcessVisualizer;