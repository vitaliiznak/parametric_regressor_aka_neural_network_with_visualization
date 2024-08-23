import { Component, createSignal } from "solid-js";
import { colors } from '../styles/colors';
import { typography } from '../styles/typography';
import { commonStyles } from '../styles/common';
import { css } from "@emotion/css";
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
    display: flex;
    align-items: flex-end;
    gap: 0.5rem;
  `,
  inputGroup: css`
    flex: 1;
  `,
  label: css`
    ${commonStyles.label}
  `,
  input: css`
    ${commonStyles.input}
  `,
  inputError: css`
    border-color: ${colors.error};
  `,
  errorMessage: css`
    color: ${colors.error};
    font-size: ${typography.fontSize.sm};
    margin-top: 0.25rem;
  `,
  button: css`
    ${commonStyles.button}
    ${commonStyles.primaryButton}
  `,
};

interface SimulationInputFormProps {
  onSimulate: (chatGPTUsage: number) => void;
}

const SimulationInputForm: Component<SimulationInputFormProps> = ({ onSimulate }) => {
  const [chatGPTUsage, setChatGPTUsage] = createSignal("");
  const [error, setError] = createSignal<string | null>(null);

  const handleSubmit = (e: Event) => {
    e.preventDefault();
    setError(null);

    const value = Number(chatGPTUsage());
    if (isNaN(value) || value < 0 || value > 100) {
      setError("Please enter a valid percentage between 0 and 100.");
      return;
    }

    onSimulate(value);
  };

  return (
    <div class={styles.container}>
      <h3 class={commonStyles.sectionTitle}>Simulate ChatGPT Usage</h3>
      <form onSubmit={handleSubmit} class={styles.form}>
        <div class={styles.inputGroup}>
          <label htmlFor="chatGPTUsage" class={commonStyles.label}>
            ChatGPT Usage (%)
          </label>
          <Tooltip content="Enter a value between 0 and 100">
            <input
              id="chatGPTUsage"
              type="number"
              min="0"
              max="100"
              step="0.1"
              value={chatGPTUsage()}
              onInput={(e) => setChatGPTUsage(e.currentTarget.value)}
              placeholder="0-100"
              class={`${commonStyles.input} ${error() ? styles.inputError : ''}`}
            />
          </Tooltip>
          {error() && <div class={styles.errorMessage}>{error()}</div>}
        </div>
        <button type="submit" class={styles.button}>
          Simulate
        </button>
      </form>
    </div>
  );
};

export default SimulationInputForm;