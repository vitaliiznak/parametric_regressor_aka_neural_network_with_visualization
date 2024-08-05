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

    if (layers.length < 2) {
      alert("Please enter at least two layers (input and output)");
      return;
    }

    if (activationsFunctions.length < layers.length ) {
      alert("The number of activation functions should be equal or larger to the number of layers");
      return;
    }

    const newNetwork = new MLP(layers[0], layers, activationsFunctions);
    console.log("New network created:", newNetwork);

    store.setState({ network: newNetwork });
    console.log("Store updated with new network");

    // Force a re-render by updating the layersString and activations
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
          Activations (comma-separated):
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
