import { Component, Show, createEffect, createSignal } from "solid-js";
import { css } from "@emotion/css";
import { VisualNode } from "../types";
import { colors } from '../styles/colors';
import { typography } from '../styles/typography';
import { FaSolidChevronDown, FaSolidChevronRight } from 'solid-icons/fa';
import { Chart } from 'chart.js/auto';

interface NeuronInfoSidebarProps {
  neuron: VisualNode | null;
  onClose: () => void;
}

const NeuronInfoSidebar: Component<NeuronInfoSidebarProps> = (props) => {
  const [expandedSections, setExpandedSections] = createSignal<Set<string>>(new Set(['basic']));
  let weightsChartInstance: Chart | null = null;
  let activationChartInstance: Chart | null = null;

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(section)) {
        newSet.delete(section);
      } else {
        newSet.add(section);
      }
      return newSet;
    });
  };

  createEffect(() => {
    if (props.neuron) {
      renderWeightsChart();
      renderActivationFunctionChart();
    }
  });

  const renderWeightsChart = () => {
    const ctx = document.getElementById('weightsChart') as HTMLCanvasElement;
    if (ctx && props.neuron) {
      if (weightsChartInstance) {
        weightsChartInstance.destroy();
      }
      weightsChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: props.neuron.weights.map((_, i) => `W${i + 1}`),
          datasets: [{
            label: 'Weights',
            data: props.neuron.weights,
            backgroundColor: colors.primary,
          }]
        },
        options: {
          responsive: true,
          scales: {
            y: {
              beginAtZero: true
            }
          }
        }
      });
    }
  };

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

  return (
    <Show when={props.neuron}>
      <div class={styles.sidebar}>
        <button onClick={props.onClose} class={styles.closeButton}>
          &times;
        </button>
        <h2 class={styles.title}>Neuron Information</h2>
        
        <Section
          title="Basic Information"
          expanded={expandedSections().has('basic')}
          onToggle={() => toggleSection('basic')}
        >
          <p><strong>ID:</strong> {props.neuron?.id}</p>
          <p><strong>Layer:</strong> {props.neuron?.layerId}</p>
          <p><strong>Activation:</strong> {props.neuron?.activation}</p>
          <p><strong>Output Value:</strong> {props.neuron?.outputValue?.toFixed(4)}</p>
        </Section>

        <Section
          title="Weights and Bias"
          expanded={expandedSections().has('weights')}
          onToggle={() => toggleSection('weights')}
        >
          <canvas id="weightsChart" width="350" height="200" style="width: 100%; height: auto;"></canvas>
          <p><strong>Bias:</strong> {props.neuron?.bias.toFixed(4)}</p>
        </Section>

        <Section
          title="Activation Function"
          expanded={expandedSections().has('activation')}
          onToggle={() => toggleSection('activation')}
        >
          <canvas id="activationFunctionChart" width="350" height="200" style="width: 100%; height: auto;"></canvas>
        </Section>

        <Section
          title="Connections"
          expanded={expandedSections().has('connections')}
          onToggle={() => toggleSection('connections')}
        >
          <p><strong>Input Connections:</strong> {props.neuron?.sourceNodes?.join(', ')}</p>
          <p><strong>Output Connections:</strong> {props.neuron?.targetNodes?.join(', ')}</p>
        </Section>

        <Section
          title="Layer Information"
          expanded={expandedSections().has('layer')}
          onToggle={() => toggleSection('layer')}
        >
          <p><strong>Layer ID:</strong> {props.neuron?.layerId}</p>
          <p><strong>Layer Activation:</strong> {props.neuron?.layerActivation}</p>
        </Section>
      </div>
    </Show>
  );
};

const Section: Component<{ title: string; expanded: boolean; onToggle: () => void; children: any }> = (props) => {
  return (
    <div class={styles.section}>
      <button class={styles.sectionToggle} onClick={props.onToggle}>
        {props.expanded ? <FaSolidChevronDown /> : <FaSolidChevronRight />}
        {props.title}
      </button>
      <Show when={props.expanded}>
        <div class={styles.sectionContent}>
          {props.children}
        </div>
      </Show>
    </div>
  );
};

const styles = {
  sidebar: css`
    position: fixed;
    top: 0;
    right: 0;
    width: 400px; // Increased from 300px to 400px
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
  section: css`
    margin-bottom: 1rem;
  `,
  sectionToggle: css`
    display: flex;
    align-items: center;
    width: 100%;
    background: none;
    border: none;
    text-align: left;
    font-size: ${typography.fontSize.lg};
    color: ${colors.text};
    cursor: pointer;
    padding: 0.5rem 0;
  `,
  sectionContent: css`
    padding-left: 1rem;
  `,
};

export default NeuronInfoSidebar;