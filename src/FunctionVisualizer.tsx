import { Component, createEffect, onMount, createSignal, onCleanup } from 'solid-js';
import Plotly from 'plotly.js-dist';
import { store } from './store';
import { getTrueFunction } from './utils/dataGeneration';
import { colors } from './styles/colors';
import { typography } from './styles/typography';
import { commonStyles, spacing } from './styles/common';
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

    // Prepare data for the neural network predictions
    const nnX = xs.map(x => x[0]);
    const nnY = ys;

    const data = [
      {
        x: trueX,
        y: trueY,
        type: 'scatter',
        mode: 'lines+text', // Include text labels
        name: 'True Function',
        line: { color: colors.primary, width: 3 },
       
        hoverinfo: 'x+y',
      },
      {
        x: nnX,
        y: nnY,
        type: 'scatter',
        mode: 'markers+text', // Include text labels
        name: 'Training Data',
        marker: { color: colors.error, size: 8 },

        hoverinfo: 'x+y',
      },
      {
        x: trueX,
        y: learnedY,
        type: 'scatter',
        mode: 'lines+text', // Include text labels
        name: 'Learned Function',
        line: { color: colors.success, width: 3, dash: 'dash' },
        visible: showLearnedFunction() ? true : 'legendonly',
  
        hoverinfo: 'x+y',
      }
    ];

    const layout = {
      xaxis: {
        title: 'ChatGPT Usage (%)',
        range: [0, 100],
        gridcolor: colors.border,
        zerolinecolor: colors.border,
      },
      yaxis: {
        title: 'Productivity Score',
        range: [0, 100],
        gridcolor: colors.border,
        zerolinecolor: colors.border,
      },
      legend: {
        x: 1,
        xanchor: 'right',
        y: 1,
   
        bordercolor: colors.border,
        borderwidth: 1,
      },
      hovermode: 'closest',
      plot_bgcolor: '#1B213D', // Ensure this matches the desired background color
      paper_bgcolor: '#1B213D',
      font: {
        family: typography.fontFamily,
        size: 14,
        color: colors.text,
      }
    };

    const config = {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToAdd: ['select2d', 'lasso2d'],
      modeBarButtonsToRemove: ['autoScale2d'],
      displaylogo: false,
      scrollZoom: true,
    };

    Plotly.newPlot(plotDiv, data, layout, config);

    // Add event listener for legend clicks
    plotDiv.on('plotly_legendclick', (event) => {
      if (event.curveNumber === 2) { // Learned Function
        setShowLearnedFunction(!showLearnedFunction());
      }
      return false; // Prevent default legend click behavior
    });
  }

  onMount(() => {
    createPlot();
    const resizeObserver = new ResizeObserver(() => {
      if (plotDiv) {
        Plotly.Plots.resize(plotDiv);
      }
    });
    if (plotDiv) {
      resizeObserver.observe(plotDiv);
    }

    onCleanup(() => {
      resizeObserver.disconnect();
    });
  });

  createEffect(() => {
    if (store.trainingData && store.network) {
      createPlot();
    }
  });

  const styles = {
    container: css`
      ${commonStyles.card}
      padding: ${spacing.xl};
      margin-top: ${spacing.xl};
      background-color: #1B213D;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
      border-radius: 8px;
      height: 100%;
      display: flex;
      flex-direction: column;
      
      @media (max-width: 768px) {
        padding: ${spacing.lg};
      }
    `,
    title: css`
      font-size: ${typography.fontSize['2xl']};
      font-weight: ${typography.fontWeight.bold};
      margin-bottom: ${spacing.lg};
      color: ${colors.text};
    `,
    plotContainer: css`
      flex-grow: 1;
      min-height: 0;
      background-color: #1B213D;
    `,
    toggleButton: css`
      ${commonStyles.button}
      ${commonStyles.secondaryButton}
      display: flex;
      align-items: center;
      justify-content: center;
      gap: ${spacing.sm};
      width: 100%;
      max-width: 200px;
      margin: ${spacing.md} auto 0;
    `,
  };

  return (
    <div class={styles.container}>
      <h2 class={styles.title}>ChatGPT Productivity Function</h2>
      <div ref={el => plotDiv = el} class={styles.plotContainer}></div>
      <button class={styles.toggleButton} onClick={() => setShowLearnedFunction(!showLearnedFunction())}>
        {showLearnedFunction() ? 'Hide' : 'Show'} Learned Function
      </button>
    </div>
  );
};

export default FunctionVisualizer;