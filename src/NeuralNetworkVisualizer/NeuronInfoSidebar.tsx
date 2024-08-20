import { Component, Show, createEffect } from "solid-js";
import { css } from "@emotion/css";
import { VisualNode } from "../types";

interface NeuronInfoSidebarProps {
  neuron: VisualNode | null;
  onClose: () => void;
}

const NeuronInfoSidebar: Component<NeuronInfoSidebarProps> = (props) => {
  createEffect(() => {
    console.log("NeuronInfoSidebar rendering with neuron:", props.neuron);
  });

  return (
    <Show when={props.neuron}>
      <div
        class={css`
          position: fixed;
          top: 0;
          right: 0;
          width: 32%;
          min-width: 250px;
          max-width: 580px;
          height: 100%;
          background-color: white;
          box-shadow: -2px 0 5px rgba(0, 0, 0, 0.1);
          transform: translateX(${props.neuron ? '0' : '100%'});
          transition: transform 0.3s ease-in-out;
          z-index: 1000;
          overflow-y: auto;
          padding: 20px;
          box-sizing: border-box;
        `}
      >
        <button
          onClick={props.onClose}
          class={css`
            position: absolute;
            top: 10px;
            right: 10px;
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
          `}
        >
          &times;
        </button>
        <h2>Neuron Information</h2>
        <div
          class={css`
            margin-top: 20px;
          `}
        >
          <p><strong>ID:</strong> {props.neuron?.id}</p>
          <p><strong>Layer:</strong> {props.neuron?.layerId}</p>
          <p><strong>Activation:</strong> {props.neuron?.activation}</p>
          <p><strong>Output Value:</strong> {props.neuron?.outputValue?.toFixed(4)}</p>
          <h3>Weights</h3>
          <ul>
            {props.neuron?.weights.map((weight, index) => (
              <li>Weight {index + 1}: {weight.toFixed(4)}</li>
            ))}
          </ul>
          <p><strong>Bias:</strong> {props.neuron?.bias.toFixed(4)}</p>
        </div>
      </div>
    </Show>
  );
};

export default NeuronInfoSidebar;