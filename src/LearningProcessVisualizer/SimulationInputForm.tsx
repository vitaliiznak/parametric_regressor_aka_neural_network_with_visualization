import { Component, createSignal } from "solid-js";
import { colors } from '../styles/colors';
import { css } from "@emotion/css";
import Tooltip from '../components/Tooltip';

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
    background-color: ${colors.surface}; /* Add background color */
    color: ${colors.text}; /* Add text color */
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
          <Tooltip content="Enter a value between 0 and 100 to represent the percentage of ChatGPT usage">
            <label htmlFor="chatGPTUsage" class={styles.label}>
              ChatGPT Usage (%)
            </label>
          </Tooltip>
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
        <Tooltip content="Run the simulation with the entered ChatGPT usage percentage">
          <button type="submit" class={styles.button}>
            Simulate
          </button>
        </Tooltip>
      </form>
    </div>
  );
};

export default SimulationInputForm;