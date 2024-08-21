import { Component, createSignal } from "solid-js";
import { colors } from '../styles/colors';
import { css } from "@emotion/css";

const styles = {
  title: css`
    font-size: 1.25rem;
    font-weight: 500;
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
    gap: 0.5rem;
  `,
  label: css`
    font-weight: 500;
    color: ${colors.textLight};
  `,
  input: css`
    border: 1px solid ${colors.border};
    border-radius: 0.25rem;
    padding: 0.5rem 0.75rem;
    font-size: 1rem;
    width: 100%;
    &:focus {
      outline: none;
      border-color: ${colors.primary};
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
    }
  `,
  inputError: css`
    border-color: ${colors.error};
  `,
  errorMessage: css`
    color: ${colors.error};
    font-size: 0.875rem;
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
      <h3 class={styles.title}>Simulate ChatGPT Usage</h3>
      <form onSubmit={handleSubmit} class={styles.form}>
        <div class={styles.inputGroup}>
          <label htmlFor="chatGPTUsage" class={styles.label}>
            ChatGPT Usage (%)
          </label>
          <input
            id="chatGPTUsage"
            type="number"
            min="0"
            max="100"
            step="0.1"
            value={chatGPTUsage()}
            onInput={(e) => setChatGPTUsage(e.currentTarget.value)}
            placeholder="Enter a value between 0 and 100"
            class={`${styles.input} ${error() ? styles.inputError : ''}`}
          />
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