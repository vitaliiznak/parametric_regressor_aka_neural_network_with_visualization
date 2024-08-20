import { Component, createSignal } from "solid-js";
import { css } from "@emotion/css";

const styles = {
  title: css`
    font-size: 1.25rem;
    font-weight: 500;
    margin-bottom: 1rem;
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
  `,
  input: css`
    border: 1px solid #ccc;
    border-radius: 0.25rem;
    padding: 0.5rem 0.75rem;
    font-size: 1rem;
    width: 100%;
  `,
  inputError: css`
    border-color: #e53e3e;
  `,
  errorMessage: css`
    color: #e53e3e;
    font-size: 0.875rem;
  `,
  button: css`
    background-color: #3b82f6;
    color: white;
    border: none;
    border-radius: 0.25rem;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    cursor: pointer;
    transition: background-color 0.2s ease-in-out;

    &:hover {
      background-color: #2563eb;
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
    <div>
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