import { Component, createSignal } from "solid-js";
import { useAppStore } from "../AppContext";

const TrainingConfigForm: Component = () => {
  const [state, setState] = useAppStore();
  const [config, setConfig] = createSignal(state.trainingConfig);

  const handleSubmit = (e: Event) => {
    e.preventDefault();
    setState({ trainingConfig: config() });
    console.log("Training config updated:", config());
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label>
          Learning Rate:
          <input
            type="number"
            step="0.001"
            value={config().learningRate}
            onInput={(e) => setConfig({ ...config(), learningRate: Number(e.currentTarget.value) })}
          />
        </label>
      </div>
      <div>
        <label>
          Epochs:
          <input
            type="number"
            value={config().epochs}
            onInput={(e) => setConfig({ ...config(), epochs: Number(e.currentTarget.value) })}
          />
        </label>
      </div>
      <div>
        <label>
          Batch Size:
          <input
            type="number"
            value={config().batchSize}
            onInput={(e) => setConfig({ ...config(), batchSize: Number(e.currentTarget.value) })}
          />
        </label>
      </div>
      <button type="submit">Update Training Config</button>
    </form>
  );
};


export default TrainingConfigForm