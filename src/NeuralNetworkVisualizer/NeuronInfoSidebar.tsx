import { Component, Show, createEffect, createSignal } from "solid-js";
import { css } from "@emotion/css";
import { VisualNode } from "../types";
import { colors } from '../styles/colors';
import { typography } from '../styles/typography';
import { Chart } from 'chart.js/auto';

interface NeuronInfoSidebarProps {
  neuron: VisualNode | null;
  onClose: () => void;
}

const NeuronInfoSidebar: Component<NeuronInfoSidebarProps> = (props) => {
  let activationChartInstance: Chart | null = null;
  const [showDetails, setShowDetails] = createSignal(false);

  createEffect(() => {
    if (props.neuron) {
      renderActivationFunctionChart();
    }
  });

  const renderActivationFunctionChart = () => {
    const ctx = document.getElementById('activationFunctionChart') as HTMLCanvasElement;
    if (ctx && props.neuron) {
      if (activationChartInstance) {
        activationChartInstance.destroy();
      }
      const x = Array.from({ length: 100 }, (_, i) => (i - 50) / 10);
      let y;
      switch (props.neuron.activation) {
        case 'tanh':
          y = x.map(Math.tanh);
          break;
        case 'relu':
          y = x.map(v => Math.max(0, v));
          break;
        case 'sigmoid':
          y = x.map(v => 1 / (1 + Math.exp(-v)));
          break;
        default:
          y = x;
      }

      activationChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels: x,
          datasets: [{
            label: `${props.neuron.activation} Activation`,
            data: y,
            borderColor: colors.primary,
            fill: false,
          }]
        },
        options: {
          responsive: true,
          scales: {
            x: {
              title: {
                display: true,
                text: 'Input'
              }
            },
            y: {
              title: {
                display: true,
                text: 'Output'
              }
            }
          }
        }
      });
    }
  };

  const calculateEquation = (neuron: VisualNode) => {
    const terms = neuron.weights.map((w, i) => `x${i + 1} * ${w.toFixed(4)}`);
    return `${terms.join(' + ')} + ${neuron.bias.toFixed(4)}`;
  };

  const calculateResult = (neuron: VisualNode): string => {
    if (!neuron.inputValues || neuron.inputValues.some(v => v === undefined)) {
      return 'N/A';
    }
    const result = neuron.weights.reduce((sum, w, i) => sum + w * (neuron.inputValues![i] || 0), neuron.bias);
    return result.toFixed(4);
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
          <p><strong>Layer:</strong> {props.neuron?.layerId}</p>
          <p><strong>Activation:</strong> {props.neuron?.activation}</p>
          <p><strong>Output:</strong> {props.neuron?.outputValue?.toFixed(4)}</p>
        </div>

        <div class={styles.equationSection}>
          <h3>Neuron Equation:</h3>
          <p>{calculateEquation(props.neuron!)} = {calculateResult(props.neuron!)}</p>
          <button onClick={() => setShowDetails(!showDetails())} class={styles.detailsButton}>
            {showDetails() ? 'Hide Details' : 'Show Details'}
          </button>
          <Show when={showDetails()}>
            <div class={styles.detailsSection}>
              {props.neuron!.weights.map((w, i) => (
                <p>x{i + 1} = {props.neuron!.inputValues?.[i]?.toFixed(4) || 'N/A'}, w{i + 1} = {w.toFixed(4)}</p>
              ))}
              <p>bias = {props.neuron!.bias.toFixed(4)}</p>
            </div>
          </Show>
        </div>

        <div class={styles.chartSection}>
          <h3>Activation Function:</h3>
          <canvas id="activationFunctionChart" width="350" height="200" style="width: 100%; height: auto;"></canvas>
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
    height: 100%;
    background-color: ${colors.surface};
    box-shadow: -2px 0 5px rgba(0, 0, 0, 0.1);
    overflow-y: auto;
    padding: 20px;
    box-sizing: border-box;
    z-index: 1000;
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
};

export default NeuronInfoSidebar;