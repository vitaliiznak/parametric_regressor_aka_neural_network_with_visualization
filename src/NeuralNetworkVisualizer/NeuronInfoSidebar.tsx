import { Component, Show, createEffect } from "solid-js";
import { css } from "@emotion/css";
import { VisualNode } from "../types";
import { colors } from '../styles/colors';
import { typography } from '../styles/typography';

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
          width: 300px; /* Fixed width */
          min-width: 250px;
          max-width: 580px;
          height: 100%;
          background-color: ${colors.surface};
          box-shadow: -2px 0 5px rgba(0, 0, 0, 0.1);
          transform: translateX(${props.neuron ? '0' : '100%'});
          transition: transform 0.3s ease-in-out;
          z-index: 1000;
          overflow-y: auto;
          padding: 20px;
          box-sizing: border-box;
          border-radius: 8px;
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
            color: ${colors.text};
          `}
        >
          &times;
        </button>
        < h2 class={css`
          color: ${colors.primary};
          font-size: ${typography.fontSize.xl};
          margin-bottom: 1rem;
        `} >Neuron Information</h2>
        <div
          class={css`
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
          `}
        >
          <div>
            < p><strong>ID:</strong> {props.neuron?.id}</p>
            < p><strong>Layer:</strong> {props.neuron?.layerId}</p>
            < p><strong>Activation:</strong> {props.neuron?.activation}</p>
            < p><strong>Output Value:</strong> {props.neuron?.outputValue?.toFixed(4)}</p>
            <h3>Weights</h3>
            <ul>
              {props.neuron?.weights.map((weight, index) => (
                <li>Weight {index + 1}: {weight.toFixed(4)}</li>
              ))}
            </ul>
            < p><strong>Bias:</strong> {props.neuron?.bias.toFixed(4)}</p>
          </div>
          <div>
            < p><strong>Input Values:</strong> {props.neuron?.inputValues?.join(', ')}</p>
            < p><strong>Gradient:</strong> {props.neuron?.gradient?.toFixed(4)}</p>
            < p><strong>Pre-Activation Value:</strong> {props.neuron?.preActivationValue?.toFixed(4)}</p>
            <h3>Detailed Weights</h3>
            < table class={css`
              width: 100%;
              border-collapse: collapse;
            `} >
              <thead>
                <tr>
                  < th class={css`
                    padding: 8px;
                    text-align: left;
                    background-color: ${colors.grayLight};
                  `} >Weight Index</th>
                  < th class={css`
                    padding: 8px;
                    text-align: left;
                    background-color: ${colors.grayLight};
                  `} >Weight Value</th>
                </tr>
              </thead>
              <tbody>
                {props.neuron?.weights.map((weight, index) => (
                  <tr key={index}>
                    < td class={css`
                      padding: 8px;
                      border-bottom: 1px solid ${colors.grayLighter};
                    `} >{index + 1}</td>
                    < td class={css`
                      padding: 8px;
                      border-bottom: 1px solid ${colors.grayLighter};
                    `} >{weight.toFixed(4)}</td >
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <h3>Connection Information</h3>
            < p><strong>Source Nodes:</strong> {props.neuron?.sourceNodes?.join(', ')}</p>
            < p><strong>Target Nodes:</strong> {props.neuron?.targetNodes?.join(', ')}</p>
          </div>
          <div>
            <h3>Layer Information</h3>
            < p><strong>Layer ID:</strong> {props.neuron?.layerId}</p>
            < p><strong>Layer Activation:</strong> {props.neuron?.layerActivation}</p>
          </div>
          <div>
            <h3>Network Information</h3>
            < p><strong>Number of Layers:</strong> {props.neuron?.network?.numLayers}</p>
            < p><strong>Total Number of Neurons:</strong> {props.neuron?.network?.totalNeurons}</p>
          </div>
          <div>
            <h3>Training Progress</h3>
            < p><strong>Weight Updates:</strong> {props.neuron?.weightUpdates?.join(', ')}</p>
          </div>
        </div>
      </div>
    </Show>
  );
};

export default NeuronInfoSidebar;