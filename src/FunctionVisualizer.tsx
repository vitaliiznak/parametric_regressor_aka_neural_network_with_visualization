import { Component, createEffect, onMount } from 'solid-js';
import Plotly from 'plotly.js-dist';
import { useAppStore } from './AppContext';
import { getTrueFunction } from './utils/dataGeneration';

const FunctionVisualizer: Component = () => {
  const [state] = useAppStore();
  let plotDiv: HTMLDivElement | undefined;

  const createPlot = () => {
    if (!plotDiv || !state.trainingData) return;

    const { xs, ys } = state.trainingData;

    // Generate points for the true function
    const trueX = Array.from({ length: 100 }, (_, i) => i);
    const trueY = trueX.map(getTrueFunction);

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
        line: { color: 'blue' }
      },
      {
        x: nnX,
        y: nnY,
        type: 'scatter',
        mode: 'markers',
        name: 'Training Data',
        marker: { color: 'red' }
      }
    ];

    const layout = {
      title: 'ChatGPT Productivity Paradox',
      xaxis: { title: 'ChatGPT Usage (%)' },
      yaxis: { title: 'Productivity Score' }
    };

    Plotly.newPlot(plotDiv, data, layout);
  };

  onMount(() => {
    createPlot();
  });

  createEffect(() => {
    if (state.trainingData) {
      createPlot();
    }
  });

  return <div ref={plotDiv} />;
};

export default FunctionVisualizer;