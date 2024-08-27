import { Component, Show, createEffect } from "solid-js";
import { css } from "@emotion/css";
import { VisualNode } from "../types";
import { colors } from '../styles/colors';
import { typography } from '../styles/typography';
import Plotly from 'plotly.js-dist';

interface NeuronInfoSidebarProps {
  neuron: VisualNode | null;
  onClose: () => void;
}

const NeuronInfoSidebar: Component<NeuronInfoSidebarProps> = (props) => {

  createEffect(() => {
    if (props.neuron) {
      renderActivationFunctionChart();
    }
  });

  const renderActivationFunctionChart = () => {
    const plotDiv = document.getElementById('activationFunctionPlot');
    if (plotDiv && props.neuron) {
      let activationFunction: (x: number) => number;
      let xMin: number, xMax: number, yMin: number, yMax: number;
      switch (props.neuron.activation) {
        case 'tanh':
          activationFunction = Math.tanh;
          xMin = -4; xMax = 4; yMin = -1; yMax = 1;
          break;
        case 'relu':
          activationFunction = (v) => Math.max(0, v);
          xMin = -2; xMax = 4; yMin = -0.5; yMax = 4;
          break;
        case 'sigmoid':
          activationFunction = (v) => 1 / (1 + Math.exp(-v));
          xMin = -6; xMax = 6; yMin = 0; yMax = 1;
          break;
        default:
          activationFunction = (v) => v;
          xMin = -4; xMax = 4; yMin = -4; yMax = 4;
      }

      const neuronInput = props.neuron.inputValues ? props.neuron.inputValues.reduce((sum, val, i) => sum + val * props.neuron!.weights[i], 0) + props.neuron.bias : 0;
      const neuronOutput = activationFunction(neuronInput);

      // Adjust x-axis limits to include neuron input
      xMin = Math.min(xMin, neuronInput - 1);
      xMax = Math.max(xMax, neuronInput + 1);

      // Adjust y-axis limits to include neuron output
      yMin = Math.min(yMin, neuronOutput - 0.5);
      yMax = Math.max(yMax, neuronOutput + 0.5);

      const x = Array.from({ length: 200 }, (_, i) => xMin + (i / 199) * (xMax - xMin));
      const y = x.map(activationFunction);

      const data = [
        {
          x: x,
          y: y,
          type: 'scatter',
          mode: 'lines',
          name: `${props.neuron.activation} Activation`,
          line: { color: colors.primary, width: 3 }
        },
        {
          x: [neuronInput],
          y: [neuronOutput],
          type: 'scatter',
          mode: 'markers',
          name: 'Neuron Input/Output',
          marker: { color: colors.error, size: 12, symbol: 'star' }
        },
        {
          x: [neuronInput, neuronInput],
          y: [yMin, neuronOutput],
          type: 'scatter',
          mode: 'lines',
          name: 'Input Line',
          line: { color: colors.secondary, dash: 'dash', width: 2 }
        },
        {
          x: [xMin, neuronInput],
          y: [neuronOutput, neuronOutput],
          type: 'scatter',
          mode: 'lines',
          name: 'Output Line',
          line: { color: colors.secondary, dash: 'dash', width: 2 }
        }
      ];

      const layout = {
        title: {
          text: 'Activation Function',
          font: { size: 24, color: colors.text }
        },
        xaxis: {
          title: 'Input',
          range: [xMin, xMax],
          gridcolor: colors.border,
          zerolinecolor: colors.border
        },
        yaxis: {
          title: 'Output',
          range: [yMin, yMax],
          gridcolor: colors.border,
          zerolinecolor: colors.border
        },
        showlegend: false,
        annotations: [
          {
            x: neuronInput,
            y: yMin,
            xref: 'x',
            yref: 'y',
            text: `Input: ${neuronInput.toFixed(4)}`,
            showarrow: true,
            arrowhead: 4,
            ax: 0,
            ay: 40
          },
          {
            x: xMin,
            y: neuronOutput,
            xref: 'x',
            yref: 'y',
            text: `Output: ${neuronOutput.toFixed(4)}`,
            showarrow: true,
            arrowhead: 4,
            ax: 40,
            ay: 0
          }
        ],
        shapes: [
          {
            type: 'circle',
            xref: 'x',
            yref: 'y',
            x0: neuronInput - 0.1,
            y0: neuronOutput - 0.1,
            x1: neuronInput + 0.1,
            y1: neuronOutput + 0.1,
            fillcolor: colors.error,
            line: { color: colors.error }
          }
        ],
        plot_bgcolor: colors.background,
        paper_bgcolor: colors.surface,
        font: { color: colors.text },
        margin: { t: 50, r: 50, b: 50, l: 50 }
      };

      const config = {
        responsive: true,
        displayModeBar: false
      };

      Plotly.newPlot(plotDiv, data, layout, config);
    }
  };

  const Term: Component<{ variable: string, value: string | number, subscript?: number }> = (props) => (
    <span class={styles.term}>
      <span class={styles.variable}>
        {props.variable}{props.subscript !== undefined && <sub>{props.subscript}</sub>}
      </span>
      {" = "}
      <span class={styles.value}>{props.value}</span>
    </span>
  );

  const calculateEquation = (neuron: VisualNode) => {
    const terms = neuron.weights.map((w, i) => {
      const x = neuron.inputValues?.[i]?.toFixed(4) || 'N/A';
      const weight = w.toFixed(4);
      return (
        <>
          (<Term variable="X" value={x} subscript={i + 1} />) *
          (<Term variable="w" value={weight} subscript={i + 1} />)
        </>
      );
    });

    const biasResult = neuron.bias.toFixed(4);
    const totalResult = neuron.outputValue?.toFixed(4) || 'N/A';

    return (
      <div class={styles.equationText}>
        {terms.map((term, index) => (
          <>
            {index > 0 && " + "}
            {term}
          </>
        ))}
        {" + "}
        (<Term variable="b" value={biasResult} />)
        {" = "}
        <span class={styles.totalResult}>{totalResult}</span>
      </div>
    );
  };

  return (
    <Show when={props.neuron}>
      <div class={styles.sidebar}>
        <button onClick={props.onClose} class={styles.closeButton}>
          &times;
        </button>
        <h2 class={styles.title}>Neuron Info</h2>
        
        <div class={styles.infoSection}>
          <p><strong>ID:</strong> {props.neuron?.id}</p>
        </div>

        <div class={styles.equationSection}>
          <h3>Neuron Equation:</h3>
          {calculateEquation(props.neuron!)}
        </div>

        <div class={styles.chartSection}>
          <div id="activationFunctionPlot" style="width: 100%; height: 300px;"></div>
        </div>
      </div>
    </Show>
  );
};

const styles = {
  sidebar: css`
    position: fixed;
    top: 0;
    right: 0;
    width: 400px;
    height: 100vh;
    background-color: ${colors.surface};
    box-shadow: -2px 0 5px rgba(0, 0, 0, 0.1);
    overflow-y: auto;
    padding: 20px;
    box-sizing: border-box;
    z-index: 1000;
    display: flex;
    flex-direction: column;
  `,
  closeButton: css`
    position: absolute;
    top: 10px;
    right: 10px;
    background: none;
    border: none;
    font-size: 1.5rem;
    cursor: pointer;
    color: ${colors.text};
  `,
  title: css`
    color: ${colors.primary};
    font-size: ${typography.fontSize.xl};
    margin-bottom: 1rem;
  `,
  infoSection: css`
    margin-bottom: 1rem;
  `,
  equationSection: css`
    margin-bottom: 1rem;
  `,
  detailsButton: css`
    background-color: ${colors.secondary};
    color: ${colors.surface};
    border: none;
    padding: 5px 10px;
    border-radius: 4px;
    cursor: pointer;
    margin-top: 10px;
  `,
  detailsSection: css`
    margin-top: 10px;
    padding: 10px;
    background-color: ${colors.background};
    border-radius: 4px;
  `,
  chartSection: css`
    margin-top: 1rem;
  `,
  equationText: css`
    font-family: monospace;
    white-space: pre-wrap;
    word-break: break-word;
    padding: 10px;
    background-color: ${colors.background};
    border-radius: 4px;
    line-height: 1.6;
  `,
  term: css`
    display: inline-block;
    margin: 0 4px;
  `,
  variable: css`
    font-weight: bold;
    color: ${colors.primary};
  `,
  value: css`
    color: ${colors.secondary};
    text-decoration: underline;
  `,
  termResult: css`
    color: ${colors.success};
    font-weight: bold;
  `,
  totalResult: css`
    color: ${colors.error};
    font-weight: bold;
    font-size: 1.1em;
  `,
};

export default NeuronInfoSidebar;