import { Component, onMount, createSignal, createEffect } from "solid-js";
import { store } from "../store";
import Plotly from 'plotly.js-dist';
import { css } from '@emotion/css';
import { colors } from '../styles/colors';

const InputDataVisualizer: Component = () => {

  const [plotRef, setPlotRef] = createSignal<HTMLDivElement | null>(null);

  const createPlot = () => {
    const plotEl = plotRef();
    if (plotEl && store.trainingData) {
      const { xs, ys } = store.trainingData;

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
          gridcolor: colors.border,
          zerolinecolor: colors.border,
        },
        yaxis: { 
          title: 'Productivity Score',
          gridcolor: colors.border,
          zerolinecolor: colors.border,
        },
        height: 600,
        margin: { l: 50, r: 50, b: 50, t: 80 },
        paper_bgcolor: colors.background,
        plot_bgcolor: colors.background,
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
    if (store.trainingData) {
      createPlot();
    }
  });

  return (
    <div class={css`
      background-color: ${colors.surface};
      border-radius: 0.5rem;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      padding: 1.5rem;
      margin-bottom: 1rem;
    `}>
      <h3 class={css`
        font-size: 1.25rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: ${colors.text};
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
        margin-top: 0.5rem;
        font-size: 0.875rem;
        color: ${colors.textLight};
      `}>
        <p>Drag to pan, scroll to zoom, or use the buttons in the top-right corner.</p>
        <p>Double-click to reset the view.</p>
      </div>
    </div>
  );
};

export default InputDataVisualizer;