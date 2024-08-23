import { Component, For } from "solid-js";
import { store } from "../store";
import { css } from "@emotion/css";

const styles = {
  container: css`
    margin-top: 1rem;
    background-color: #f3f4f6;
    border-radius: 0.5rem;
    padding: 1rem;
  `,
  title: css`
    font-size: 1rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
  `,
  weightList: css`
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 0.5rem;
  `,
  weightItem: css`
    background-color: white;
    border-radius: 0.25rem;
    padding: 0.5rem;
    font-size: 0.875rem;
  `,
};

const WeightUpdateVisualizer: Component = () => {
  return (
    <div class={styles.container}>
      <h4 class={styles.title}>Weight Updates</h4>
      <div class={styles.weightList}>
        <For each={store.trainingStepResult.oldWeights}>
          {(oldWeight, index) => (
            <div class={styles.weightItem}>
              Weight {index()}: 
              {oldWeight.toFixed(4)} → {store.trainingStepResult.newWeights[index()].toFixed(4)}
              (Δ: {(store.trainingStepResult.newWeights[index()] - oldWeight).toFixed(4)})
            </div>
          )}
        </For>
      </div>
    </div>
  );
};

export default WeightUpdateVisualizer;