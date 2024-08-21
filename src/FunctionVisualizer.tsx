import { Component, createEffect, onMount, createSignal } from 'solid-js';
import Plotly from 'plotly.js-dist';
import { store } from './store';
import { getTrueFunction } from './utils/dataGeneration';
import { colors } from './styles/colors';
import { css } from '@emotion/css';

const FunctionVisualizer: Component = () => {
  let plotDiv: HTMLDivElement | undefined;
  const [showLearnedFunction, setShowLearnedFunction] = createSignal(true);

  const createPlot = () => {
    if (!plotDiv || !store.trainingData || !store.network) return;

    const { xs, ys } = store.trainingData;

    // Generate points for the true function
    const trueX = Array.from({ length: 100 }, (_, i) => i);
    const trueY = trueX.map(getTrueFunction);

    // Generate points for the learned function
    const learnedY = trueX.map(x => {
      // Assuming store.network.forward accepts a regular number input
      // and returns a regular number output
      const output = store.network.forward([x]);
      return output[0].data;
    });

    console.log(learnedY)

    // Prepare data for the neural network predictions
    const nnX = xs.map(x => x[0]);
    const nnY = ys;

    const data = [
      {
        x: trueX,
        y: trueY,
        type: 'scatter',
        mode: 'lines',
        name: 'True Function',
        line: { color: colors.primary }
      },
      {
        x: nnX,
        y: nnY,
        type: 'scatter',
        mode: 'markers',
        name: 'Training Data',
        marker: { color: colors.error }
      },
      {
        x: trueX,
        y: learnedY,
        type: 'scatter',
        mode: 'lines',
        name: 'Learned Function',
        line: { color: colors.success, dash: 'dash' },
        visible: showLearnedFunction() ? true : 'legendonly'
      }
    ];

    const layout = {
      title: 'ChatGPT Productivity Paradox',
      xaxis: { title: 'ChatGPT Usage (%)' },
      yaxis: { title: 'Productivity Score' },
      legend: { x: 1, xanchor: 'right', y: 1 },
      updatemenus: []
    };

    Plotly.newPlot(plotDiv, data, layout);
  };

  onMount(() => {
    createPlot();
  });

  createEffect(() => {
    if (store.trainingData && store.network) {
      createPlot();
    }
  });

  const styles = {
    container: css`
      background-color: ${colors.surface};
      padding: 1.5rem;
      border-radius: 0.5rem;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      margin-top: 1rem;
      
      @media (max-width: 768px) {
        padding: 1rem;
      }
    `,
    title: css`
      font-size: 1.25rem;
      font-weight: bold;
      margin-bottom: 1rem;
      color: ${colors.text};
    `,
    plotContainer: css`
      width: 100%;
      height: 0;
      padding-bottom: 75%; // 4:3 aspect ratio
      position: relative;
      
      @media (max-width: 768px) {
        padding-bottom: 100%; // 1:1 aspect ratio on smaller screens
      }
    `,
    toggleButton: css`
      background-color: ${colors.secondary};
      color: ${colors.surface};
      border: none;
      border-radius: 0.25rem;
      padding: 0.5rem 1rem;
      font-size: 1rem;
      cursor: pointer;
      transition: background-color 0.2s;
      margin-top: 1rem;
      &:hover {
        background-color: ${colors.secondaryDark};
      }
    `,
  };

  return (
    <div class={styles.container}>
      <h3 class={styles.title}>Function Visualization</h3>
      <div ref={plotDiv} class={styles.plotContainer}></div>
      <button class={styles.toggleButton} onClick={() => setShowLearnedFunction(!showLearnedFunction())}>
        {showLearnedFunction() ? 'Hide' : 'Show'} Learned Function
      </button>
    </div>
  );
};

export default FunctionVisualizer;