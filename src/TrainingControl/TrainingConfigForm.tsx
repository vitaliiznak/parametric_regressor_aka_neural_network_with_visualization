import { Component } from "solid-js";
import { useAppStore } from "../AppContext";

const TrainingConfigForm: Component = () => {
  const [state, setState] = useAppStore();

  const handleSubmit = (e: Event) => {
    e.preventDefault();
    console.log("Training config updated:", state.trainingConfig);
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label>
          Learning Rate:
          <input
            type="number"
            step="0.001"
            value={state.trainingConfig.learningRate}
            onInput={(e) => setState('trainingConfig', 'learningRate', Number(e.currentTarget.value))}
          />
        </label>
      </div>
      <div>
        <label>
          Epochs:
          <input
            type="number"
            value={state.trainingConfig.epochs}
            onInput={(e) => setState('trainingConfig', 'epochs', Number(e.currentTarget.value))}
          />
        </label>
      </div>
      <div>
        <label>
          Batch Size:
          <input
            type="number"
            value={state.trainingConfig.batchSize}
            onInput={(e) => setState('trainingConfig', 'batchSize', Number(e.currentTarget.value))}
          />
        </label>
      </div>
      <button type="submit">Update Training Config</button>
    </form>
  );
};


export default TrainingConfigForm