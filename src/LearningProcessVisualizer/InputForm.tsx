import { Component, createSignal } from "solid-js";
import { useAppStore } from "../AppContext";

const InputForm: Component = () => {
  const [state, setState] = useAppStore();
  const [chatGPTUsage, setChatGPTUsage] = createSignal("");

  const handleSubmit = (e: Event) => {
    e.preventDefault();
    const value = Number(chatGPTUsage());
    if (isNaN(value) || value < 0 || value > 100) {
      alert("Please provide a valid percentage between 0 and 100");
      return;
    }
    setState('currentInput', [value]);
  };

  return (
    <form onSubmit={handleSubmit}>
      <label>
        ChatGPT Usage (%):
        <input 
          type="number" 
          min="0" 
          max="100" 
          step="0.1" 
          value={chatGPTUsage()} 
          onInput={(e) => setChatGPTUsage(e.currentTarget.value)} 
        />
      </label>
      <button type="submit">Set Input</button>
    </form>
  );
};

export default InputForm;