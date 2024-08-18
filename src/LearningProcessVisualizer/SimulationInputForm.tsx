import { Component, createSignal } from "solid-js";
import { setStore } from "../store";
import { css } from "@emotion/css";

const styles = {
  container: css`
    background-color: white;
    padding: 1.5rem;
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-top: 1rem;
  `,
  title: css`
    font-size: 1.25rem;
    font-weight: bold;
    margin-bottom: 1rem;
    color: #2c3e50;
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
    color: #4b5563;
    margin-bottom: 0.25rem;
  `,
  input: css`
    padding: 0.5rem;
    border: 1px solid #d1d5db;
    border-radius: 0.25rem;
    font-size: 1rem;
    &:focus {
      outline: none;
      border-color: #3b82f6;
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
    }
  `,
  button: css`
    background-color: #3b82f6;
    color: white;
    border: none;
    border-radius: 0.25rem;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    cursor: pointer;
    transition: background-color 0.2s;
    &:hover {
      background-color: #2563eb;
    }
  `,
  error: css`
    color: #cc0000;
    font-size: 0.875rem;
    margin-bottom: 0.5rem;
  `,
};

interface SimulationInputFormProps {
  onSimulateNetwork: () => void;
}

const SimulationInputForm: Component<SimulationInputFormProps> = ({ onSimulateNetwork }) => {
  const [chatGPTUsage, setChatGPTUsage] = createSignal("");
  const [error, setError] = createSignal<string | null>(null);

  const handleSubmit = (e: Event) => {
    e.preventDefault();
    setError(null);

    const value = Number(chatGPTUsage());
    if (isNaN(value)) {
      setError("Please enter a valid number.");
      return;
    }
    if (value < 0 || value > 100) {
      setError("Please provide a percentage between 0 and 100.");
      return;
    }

    try {
      // setStore('simulationOutput', null); // Reset simulation output
      setStore('currentInput', [value]);
      onSimulateNetwork();
    } catch (err) {
      console.error("Error during simulation:", err);
      setError("An error occurred during simulation. Please try again.");
    }
  };

  return (
    <div class={styles.container}>
      <h3 class={styles.title}>Simulation Input</h3>
      <form onSubmit={handleSubmit} class={styles.form}>
        <div class={styles.inputGroup}>
          <label class={styles.label} for="chatGPTUsage">
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
            class={styles.input}
            placeholder="Enter a value between 0 and 100"
          />
        </div>
        {error() && <div class={styles.error}>{error()}</div>}
        <div>
          <button type="submit" class={styles.button}>Set Input</button>
        </div>
      </form>
    </div>
  );
};

export default SimulationInputForm;