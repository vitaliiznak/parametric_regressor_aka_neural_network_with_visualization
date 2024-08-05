import { Component, createSignal } from "solid-js";
import { AppState, Store } from "../store";

const TrainingConfigForm: Component<{ store: Store<AppState> }> = (props) => {
  const [config, setConfig] = createSignal(props.store.getState().trainingConfig);

  const handleSubmit = (e: Event) => {
    e.preventDefault();
    props.store.setState({ trainingConfig: config() });
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