import { Component, createSignal } from "solid-js";
import { useAppStore } from "../AppContext";

const InputForm: Component = () => {
  const [state, setState] = useAppStore();
  const [inputValues, setInputValues] = createSignal<string>("");

  const handleSubmit = (e: Event) => {
    e.preventDefault();
    const values = inputValues().split(',').map(Number);
    if (values.length !== state.network.layers[0]?.neurons.length) {
      alert(`Please provide ${state.network.layers[0]?.neurons.length} input values`);
      return;
    }
    setState('currentInput', values);
  };

  return (
    <form onSubmit={handleSubmit}>
      <label>
        Input Values (comma-separated):
        <input
          type="text"
          value={inputValues()}
          onInput={(e) => setInputValues(e.currentTarget.value)}
        />
      </label>
      <button type="submit">Set Input</button>
    </form>
  );
};

export default InputForm;