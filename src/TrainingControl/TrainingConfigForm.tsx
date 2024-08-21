import { Component } from "solid-js";
import { setStore, store } from "../store";
import { css } from "@emotion/css";
import { colors } from '../styles/colors';

const styles = {
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
  form: css`
    display: flex;
    flex-direction: column;
    gap: 1rem;
  `,
  inputGroup: css`
    display: flex;
    flex-direction: column;
  `,
  label: css`
    font-size: 0.875rem;
    color: ${colors.textLight};
    margin-bottom: 0.25rem;
  `,
  input: css`
    padding: 0.5rem;
    border: 1px solid ${colors.border};
    border-radius: 0.25rem;
    font-size: 1rem;
    &:focus {
      outline: none;
      border-color: ${colors.primary};
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
    }
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
  `,
};

const TrainingConfigForm: Component = () => {
  const handleSubmit = (e: Event) => {
    e.preventDefault();
    console.log("Training config updated:", store.trainingConfig);
  };

  return (
    <div class={styles.container}>
      <h3 class={styles.title}>Training Configuration</h3>
      <form onSubmit={handleSubmit} class={styles.form}>
        <div class={styles.inputGroup}>
          <label class={styles.label}>
            Learning Rate:
            <input
              type="number"
              step="0.001"
              value={store.trainingConfig.learningRate}
              onInput={(e) => setStore('trainingConfig', 'learningRate', Number(e.currentTarget.value))}
              class={styles.input}
            />
          </label>
        </div>
        <div class={styles.inputGroup}>
          <label class={styles.label}>
            Iterations:
            <input
              type="number"
              value={store.trainingConfig.iterations}
              onInput={(e) => setStore('trainingConfig', 'iterations', Number(e.currentTarget.value))}
              class={styles.input}
            />
          </label>
        </div>
        <div class={styles.inputGroup}>
          <label class={styles.label}>
            Batch Size:
            <input
              type="number"
              value={store.trainingConfig.batchSize}
              onInput={(e) => setStore('trainingConfig', 'batchSize', Number(e.currentTarget.value))}
              class={styles.input}
            />
          </label>
        </div>
        <button type="submit" class={styles.button}>Update Training Config</button>
      </form>
    </div>
  );
};

export default TrainingConfigForm;