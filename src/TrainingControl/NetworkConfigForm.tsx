import { Component, createEffect, createSignal } from "solid-js";
import { ActivationFunction } from "../NeuralNetwork/types";
import { actions, store } from "../store";
import { colors } from '../styles/colors';
import { commonStyles } from '../styles/common';
import { typography } from '../styles/typography';
import Tooltip from '../components/Tooltip';
import { css } from "@emotion/css";

const styles = {
  container: css`
    ${commonStyles.card}
    max-width: 100%; // Ensure the container doesn't exceed its parent's width
  `,
  title: css`
    font-size: ${typography.fontSize.xl};
    font-weight: ${typography.fontWeight.bold};
    margin-bottom: 1rem;
    color: ${colors.text};
  `,
  form: css`
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  `,
  inputGroup: css`
    display: flex;
    flex-direction: column;
  `,
  label: css`
    ${commonStyles.label}
  `,
  input: css`
    ${commonStyles.input}
    width: 100%; // Make inputs take full width of their container
  `,
  button: css`
    ${commonStyles.button}
    ${commonStyles.primaryButton}
    width: 100%; // Make button take full width
  `,
  currentConfig: css`
    margin-top: 1rem;
    font-size: ${typography.fontSize.sm};
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

    if (activationsFunctions.length !== layers.length) {
      alert("The number of activation functions should be equal to the number of layers");
      return;
    }

    actions.updateNetworkConfig(layers, activationsFunctions);

    setLayersString(layers.join(','));
    setActivations(activationsFunctions.join(','));
  };

  return (
    <div class={styles.container}>
      <form class={styles.form} onSubmit={handleSubmit}>
        <div class={styles.inputGroup}>
          <label class={commonStyles.label}>
            Layer sizes:
            <Tooltip content="Enter the number of neurons for each layer, separated by commas. E.g., 5,3,1">
              <input
                type="text"
                value={layersString()}
                onInput={handleLayersChange}
                class={styles.input}
                placeholder="e.g., 5,3,1"
              />
            </Tooltip>
          </label>
        </div>
        <div class={styles.inputGroup}>
          <label class={commonStyles.label}>
            Activations:
            <Tooltip content="Enter the activation function for each layer, separated by commas. E.g., tanh,relu,sigmoid">
              <input
                type="text"
                value={activations()}
                onInput={handleActivationsChange}
                class={styles.input}
                placeholder="e.g., tanh,relu,sigmoid"
              />
            </Tooltip>
          </label>
        </div>
        <button type="submit" class={styles.button}>Update Network</button>
      </form>
      <div class={styles.currentConfig}>
        <p>Current: {store.network.layers.map(layer => layer.neurons.length).join(', ')} | {store.network.activations.join(', ')}</p>
      </div>
    </div>
  );
};

export default NetworkConfigForm;