import { Component, createSignal } from "solid-js";
import { AppState, Store } from "../store";
import { ActivationFunction } from "../NeuralNetwork/types";
import { MLP } from "../NeuralNetwork/mlp";


const NetworkConfigForm: Component<{ store: Store<AppState> }> = (props) => {
  const [layers, setLayers] = createSignal([3, 4, 1]);
  const [activations, setActivations] = createSignal(['tanh', 'tanh', 'tanh']);

  const handleSubmit = (e: Event) => {
    e.preventDefault();
    const newNetwork = new MLP(1, layers(), activations() as ActivationFunction[]);
    props.store.setState({ network: newNetwork });
    console.log("Network updated:", newNetwork);
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label>
          Layers (comma-separated):
          <input
            type="text"
            value={layers().join(',')}
            onInput={(e) => setLayers(e.currentTarget.value.split(',').map(Number))}
          />
        </label>
      </div>
      <div>
        <label>
          Activations (comma-separated):
          <input
            type="text"
            value={activations().join(',')}
            onInput={(e) => setActivations(e.currentTarget.value.split(','))}
          />
        </label>
      </div>
      <button type="submit">Update Network</button>
    </form>
  );
};



export default NetworkConfigForm
