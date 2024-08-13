import { Component, onMount, createSignal, createEffect } from "solid-js";
import { useAppStore } from "../AppContext";
import Plotly from 'plotly.js';
import { css } from '@emotion/css';

const InputDataVisualizer: Component = () => {
  const [state] = useAppStore();
  const [plotRef, setPlotRef] = createSignal<HTMLDivElement | null>(null);

  const createPlot = () => {
    const plotEl = plotRef();
    if (plotEl && state.trainingData) {
      const { xs, ys } = state.trainingData;

      const trace = {
        x: xs.map(x => x[0]),  // ChatGPT usage percentage
        y: ys,                 // Productivity score
        mode: 'markers',
        type: 'scatter',
        marker: {
          size: 10,
          color: ys,
          colorscale: 'Viridis',
          colorbar: {
            title: 'Productivity Score',
            thickness: 20,
            len: 0.5,
          },
          symbol: 'circle',
        },
        text: xs.map((x, i) => `ChatGPT Usage: ${x[0].toFixed(2)}%<br>Productivity: ${ys[i].toFixed(2)}`),
        hoverinfo: 'text'
      };

      const layout = {
        title: {
          text: 'Developer Productivity vs ChatGPT Usage',
          font: { size: 24 }
        },
        xaxis: { 
          title: 'ChatGPT Usage (%)',
          gridcolor: 'lightgray',
          zerolinecolor: 'lightgray',
        },
        yaxis: { 
          title: 'Productivity Score',
          gridcolor: 'lightgray',
          zerolinecolor: 'lightgray',
        },
        height: 600,
        margin: { l: 50, r: 50, b: 50, t: 80 },
        paper_bgcolor: 'rgb(250, 250, 250)',
        plot_bgcolor: 'rgb(250, 250, 250)',
        hovermode: 'closest',
        dragmode: 'pan',
      };

      const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToAdd: ['select2d', 'lasso2d'],
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false,
        scrollZoom: true, // Enable scroll zoom
      };

      Plotly.newPlot(plotEl, [trace], layout, config);

      // Add double-click event for resetting the zoom
      plotEl.on('plotly_doubleclick', () => {
        Plotly.relayout(plotEl, {
          'xaxis.autorange': true,
          'yaxis.autorange': true
        });
      });
    }
  };

  onMount(() => {
    createPlot();
  });

  createEffect(() => {
    if (state.trainingData) {
      createPlot();
    }
  });

  return (
    <div class={css`
      background-color: rgb(250, 250, 250);
      border-radius: 8px;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      padding: 20px;
      margin-bottom: 20px;
    `}>
      <h3 class={css`
        font-size: 1.5em;
        margin-bottom: 15px;
        color: #333;
      `}>Input Data Visualization</h3>
      <div 
        ref={setPlotRef} 
        class={css`
          width: 100%;
          height: 600px;
        `}
      ></div>
      <div class={css`
        text-align: center;
        margin-top: 10px;
        font-size: 14px;
        color: #666;
      `}>
        <p>Drag to pan, scroll to zoom, or use the buttons in the top-right corner.</p>
        <p>Double-click to reset the view.</p>
      </div>
    </div>
  );
};

export default InputDataVisualizer;