import { Component } from "solid-js";
import { setStore, store } from "../store";
import { css } from "@emotion/css";
import { colors } from '../styles/colors';
import { commonStyles } from '../styles/common';
import { typography } from '../styles/typography';
import Tooltip from '../components/Tooltip';

const styles = {
  container: css`
    ${commonStyles.card}
  `,
  title: css`
    font-size: ${typography.fontSize.xl};
    font-weight: ${typography.fontWeight.bold};
    margin-bottom: 1rem;
    color: ${colors.text};
  `,
  form: css`
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.5rem;
  `,
  inputGroup: css`
    display: flex;
    flex-direction: column;
  `,
  label: css`
    ${commonStyles.label}
  `,
  input: css`
    ${commonStyles.input}
  `,
  button: css`
    ${commonStyles.button}
    ${commonStyles.primaryButton}
    grid-column: span 2;
  `,
};

const TrainingConfigForm: Component = () => {
  const handleSubmit = (e: Event) => {
    e.preventDefault();
    console.log("Training config updated:", store.trainingConfig);
  };

  return (
    <div class={styles.container}>
      <h2 class={styles.title}>Training Configuration</h2>
      <form onSubmit={handleSubmit} class={styles.form}>
        <div class={styles.inputGroup}>
          <label class={commonStyles.label}>
            Learning Rate:
            <Tooltip content="The learning rate determines how quickly the model adapts to the problem. Typical values range from 0.001 to 0.1.">
              <input
                type="number"
                step="0.001"
                min="0.001"
                max="1"
                value={store.trainingConfig.learningRate}
                onInput={(e) => setStore('trainingConfig', 'learningRate', Number(e.currentTarget.value))}
                class={commonStyles.input}
              />
            </Tooltip>
          </label>
        </div>
        <div class={styles.inputGroup}>
          <label class={commonStyles.label}>
            Iterations:
            <Tooltip content="The number of times the entire dataset is passed through the network during training.">
              <input
                type="number"
                min="1"
                value={store.trainingConfig.iterations}
                onInput={(e) => setStore('trainingConfig', 'iterations', Number(e.currentTarget.value))}
                class={commonStyles.input}
              />
            </Tooltip>
          </label>
        </div>
        <button type="submit" class={styles.button}>Update Training Config</button>
      </form>
    </div>
  );
};

export default TrainingConfigForm;