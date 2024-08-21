import { Component, createEffect, createSignal } from "solid-js";
import { ActivationFunction } from "../NeuralNetwork/types";
import { MLP } from "../NeuralNetwork/mlp";
import { setStore, store } from "../store";
import { css } from "@emotion/css";
import { CONFIG } from "../config";
import { colors } from '../styles/colors';

const styles = {
  container: css`
    background-color: ${colors.surface};
    padding: 1.5rem;
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-top: 1rem;
  `,
  title: css`
    font-size: 1.25rem;
    font-weight: bold;
    margin-bottom: 1rem;
    color: ${colors.text};
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
    color: ${colors.textLight};
    margin-bottom: 0.25rem;
  `,
  input: css`
    padding: 0.5rem;
    border: 1px solid ${colors.border};
    border-radius: 0.25rem;
    font-size: 1rem;
    &:focus {
      outline: none;
      border-color: ${colors.primary};
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
    }
  `,
  button: css`
    background-color: ${colors.primary};
    color: ${colors.surface};
    border: none;
    border-radius: 0.25rem;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    cursor: pointer;
    transition: background-color 0.2s;
    &:hover {
      background-color: ${colors.primaryDark};
    }
  `,
  currentConfig: css`
    margin-top: 1rem;
    font-size: 0.875rem;
    color: ${colors.textLight};
  `,
};

const NetworkConfigForm: Component = () => {
  const [layersString, setLayersString] = createSignal(
    store.network.layers.map(layer => layer.neurons.length).join(',')
  );
  const [activations, setActivations] = createSignal(store.network.activations.join(','));

  createEffect(() => {
    console.log("Current network store:", store.network);
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
    
    if (layers.length === 0) {
      alert("Please enter at least one layer size");
      return;
    }

    const inputSize = CONFIG.INITIAL_NETWORK.inputSize;

    if (activationsFunctions.length !== layers.length) {
      alert("The number of activation functions should be equal to the number of layers");
      return;
    }

    const newNetwork = new MLP({
      inputSize: inputSize,
      layers: layers,
      activations: activationsFunctions
    });

    setStore({ network: newNetwork });
    console.log("Store updated with new network");

    setLayersString(layers.join(','));
    setActivations(activationsFunctions.join(','));
  };

  return (
    <div class={styles.container}>
      <h3 class={styles.title}>Network Configuration</h3>
      <form onSubmit={handleSubmit} class={styles.form}>
        <div class={styles.inputGroup}>
          <label class={styles.label}>
            Layers (comma-separated):
            <input
              type="text"
              value={layersString()}
              onInput={handleLayersChange}
              class={styles.input}
              placeholder="e.g., 5,3,1"
            />
          </label>
        </div>
        <div class={styles.inputGroup}>
          <label class={styles.label}>
            Activations (comma-separated):
            <input
              type="text"
              value={activations()}
              onInput={handleActivationsChange}
              class={styles.input}
              placeholder="e.g., tanh,relu,sigmoid"
            />
          </label>
        </div>
        <button type="submit" class={styles.button}>Update Network</button>
      </form>
      <div class={styles.currentConfig}>
        <p>Current Configuration:</p>
        <p>Layers: {store.network.layers.map(layer => layer.neurons.length).join(', ')}</p>
        <p>Activations: {store.network.activations.join(', ')}</p>
      </div>
    </div>
  );
};

export default NetworkConfigForm;