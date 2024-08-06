import { Component, createEffect, createSignal } from "solid-js";
import { ActivationFunction } from "../NeuralNetwork/types";
import { MLP } from "../NeuralNetwork/mlp";
import { useAppStore } from "../AppContext";


const NetworkConfigForm: Component = () => {
  const store = useAppStore();
  const [layersString, setLayersString] = createSignal(
    store.getState().network.layers.map(layer => layer.neurons.length).join(',')
  );
  const [activations, setActivations] = createSignal(store.getState().network.activations.join(','));

  createEffect(() => {
    console.log("Current network state:", store.getState().network);
  });

  const handleLayersChange = (e: Event) => {
    const target = e.target as HTMLInputElement;
    setLayersString(target.value);
  };

  const handleActivationsChange = (e: Event) => {
    const target = e.target as HTMLInputElement;
    setActivations(target.value);
  };

  const handleSubmit = (e: Event) => {
    e.preventDefault();
    const layers = layersString().split(',').map(Number).filter(n => !isNaN(n));
    const activationsFunctions = activations().split(',') as ActivationFunction[];
    
    console.log("Parsed layers:", layers);
    console.log("Parsed activations:", activationsFunctions);

    if (layers.length === 0) {
      alert("Please enter at least one layer size");
      return;
    }

    const inputSize = store.getState().network.layers[0].neurons.length;

    if (activationsFunctions.length !== layers.length - 1) {
      alert("The number of activation functions should be equal to the number of layers (excluding the input layer)");
      return;
    }

    // Add 'linear' activation for the input layer
    const allActivations = ['linear', ...activationsFunctions] as ActivationFunction[];

    const newNetwork = new MLP({
      inputSize: inputSize,
      layers: layers,
      activations: allActivations
    });
    console.log("New network created:", newNetwork);

    store.setState({ network: newNetwork });
    console.log("Store updated with new network");

    setLayersString(layers.join(','));
    setActivations(activationsFunctions.join(','));
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label>
          Layers (comma-separated):
          <input
            type="text"
            value={layersString()}
            onInput={handleLayersChange}
          />
        </label>
      </div>
      <div>
        <label>
          Activations (comma-separated, one for each layer except input):
          <input
            type="text"
            value={activations()}
            onInput={handleActivationsChange}
          />
        </label>
      </div>
      <button type="submit">Update Network</button>
    </form>
  );
};





export default NetworkConfigForm