import { Component, createEffect, onMount, createSignal } from 'solid-js';
import Plotly from 'plotly.js-dist';
import { store } from './store';
import { getTrueFunction } from './utils/dataGeneration';

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
        line: { color: 'blue' }
      },
      {
        x: nnX,
        y: nnY,
        type: 'scatter',
        mode: 'markers',
        name: 'Training Data',
        marker: { color: 'red' }
      },
      {
        x: trueX,
        y: learnedY,
        type: 'scatter',
        mode: 'lines',
        name: 'Learned Function',
        line: { color: 'green', dash: 'dash' },
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

  return (
    <div>
      <div ref={plotDiv} />
      <button onClick={() => {
        setShowLearnedFunction(!showLearnedFunction());
        createPlot();
      }}>
        {showLearnedFunction() ? 'Hide' : 'Show'} Learned Function
      </button>
    </div>
  );
};

export default FunctionVisualizer;