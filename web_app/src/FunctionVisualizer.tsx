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
  const [showTrueFunction, setShowTrueFunction] = createSignal(true);
  const [showLearnedFunction, setShowLearnedFunction] = createSignal(true);
  const [showTrainingData, setShowTrainingData] = createSignal(true);

  const generateTrueFunctionPoints = () => {
    const trueX: number[] = Array.from({ length: 100 }, (_, i) => i/100);
    const trueY: number[] = trueX.map(D => getTrueFunction(D));
    return { trueX, trueY };
  };

  const preparePlotData = (
    trueX: number[],
    trueY: number[],
    learnedY: number[],
    xs: number[][],
    ys: number[]
  ) => [
    {
      x: trueX,
      y: trueY,
      type: 'scatter',
      mode: 'lines',
      name: 'True Function',
      line: { color: colors.primary, width: 3 },
      visible: showTrueFunction() ? true : 'legendonly',
    },
    {
      x: xs.map(x => x[0]),
      y: ys,
      type: 'scatter',
      mode: 'markers',
      name: 'Training Data',
      marker: { color: colors.error, size: 6 },
      visible: showTrainingData() ? true : 'legendonly',
    },
    {
      x: trueX,
      y: learnedY,
      type: 'scatter',
      mode: 'lines',
      name: 'Learned Function',
      line: { color: colors.success, width: 3, dash: 'dash' },
      visible: showLearnedFunction() ? true : 'legendonly',
    },
  ];

  const createPlot = () => {
    if (!plotDiv || !store.trainingData || !store.network) return;

    const { xs, ys } = store.trainingData;
    const { trueX, trueY } = generateTrueFunctionPoints();

    const learnedY: number[] = xs.map(x => store.network.forward([x[0]])[0].data);

    const data = preparePlotData(trueX, trueY, learnedY, xs, ys);

    // Dynamically calculate axis ranges based on data
    const allX = [...trueX, ...xs.map(x => x[0])];
    const allY = [...trueY, ...learnedY, ...ys];

    const xMin = Math.min(...allX) - 5;
    const xMax = Math.max(...allX) + 5;
    const yMin = Math.min(...allY) - 0.1;
    const yMax = Math.max(...allY) + 0.1;

    const layout: Partial<Plotly.Layout> = {
      xaxis: {
        title: 'Drug Dosage (mg)',
        range: [xMin, xMax],
        gridcolor: colors.border,
        zerolinecolor: colors.border,
        tickformat: '.0f',
      },
      yaxis: {
        title: 'Immune Response',
        range: [yMin, yMax],
        gridcolor: colors.border,
        zerolinecolor: colors.border,
        tickformat: '.2f',
      },
      legend: {
        x: 1,
        xanchor: 'right',
        y: 1,
        bordercolor: colors.border,
        borderwidth: 1,
      },
      hovermode: 'closest',
      plot_bgcolor: colors.background,
      paper_bgcolor: colors.background,
      font: {
        family: typography.fontFamily,
        size: 14,
        color: colors.text,
      },
      autosize: true,
    };

    const config: Partial<Plotly.Config> = {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToAdd: ['select2d', 'lasso2d'],
      modeBarButtonsToRemove: ['autoScale2d'],
      displaylogo: false,
      scrollZoom: true,
    };

    Plotly.react(plotDiv, data, layout, config);
  };

  const handleLegendClick = (event: Plotly.PlotPlotlyClickEvent) => {
    if (event.curveNumber === 0) { // True Function
      setShowTrueFunction(prev => !prev());
    }
    if (event.curveNumber === 1) { // Training Data
      setShowTrainingData(prev => !prev());
    }
    if (event.curveNumber === 2) { // Learned Function
      setShowLearnedFunction(prev => !prev());
    }
    return false; // Prevent default behavior
  };

  onMount(() => {
    createPlot();

    const resizeObserver = new ResizeObserver(() => {
      if (plotDiv) {
        Plotly.Plots.resize(plotDiv);
      }
    });

    if (plotDiv) {
      resizeObserver.observe(plotDiv);
      plotDiv.on('plotly_legendclick', handleLegendClick);
    }

    onCleanup(() => {
      if (plotDiv) {
        Plotly.purge(plotDiv);
        plotDiv.removeEventListener('plotly_legendclick', handleLegendClick);
      }
      resizeObserver.disconnect();
    });
  });

  createEffect(() => {
    if (store.trainingData && store.network) {
      createPlot();
    }
  });

  createEffect(() => {
    if (showTrueFunction() || showLearnedFunction() || showTrainingData()) {
      createPlot();
    }
  });

  const styles = {
    container: css`
      ${commonStyles.card}
      padding: ${spacing.xl};
      margin-top: ${spacing.xl};
      background-color: ${colors.background};
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
      background-color: ${colors.background};
    `,
    toggleButtonsContainer: css`
      display: flex;
      justify-content: center;
      gap: ${spacing.sm};
      margin-top: ${spacing.md};
      flex-wrap: wrap;
    `,
    toggleButton: css`
      ${commonStyles.button}
      ${commonStyles.secondaryButton}
      display: flex;
      align-items: center;
      justify-content: center;
      gap: ${spacing.sm};
      width: 200px;
      @media (max-width: 500px) {
        width: 100%;
      }
    `,
  };

  return (
    <div class={styles.container}>
      <h2 class={styles.title}>Drug Dosage vs Immune Response</h2>
      <div ref={el => (plotDiv = el)} class={styles.plotContainer}></div>
      <div class={styles.toggleButtonsContainer}>
        <button
          class={styles.toggleButton}
          onClick={() => setShowTrueFunction(prev => !prev())}
        >
          {showTrueFunction() ? 'Hide' : 'Show'} True Function
        </button>
        <button
          class={styles.toggleButton}
          onClick={() => setShowLearnedFunction(prev => !prev())}
        >
          {showLearnedFunction() ? 'Hide' : 'Show'} Learned Function
        </button>
        <button
          class={styles.toggleButton}
          onClick={() => setShowTrainingData(prev => !prev())}
        >
          {showTrainingData() ? 'Hide' : 'Show'} Training Data
        </button>
      </div>
    </div>
  );
};

export default FunctionVisualizer;