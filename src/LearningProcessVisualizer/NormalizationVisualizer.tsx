import { Component, createEffect, createMemo } from "solid-js";
import { store } from "../store";
import Plotly from 'plotly.js-dist';
import { css } from '@emotion/css';
import { colors } from '../styles/colors';

const styles = {
  container: css`
    padding: 1rem;
    background-color: ${colors.surface};
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
  `,
  plot: css`
    width: 100%;
    height: 400px;
  `,
};

const NormalizationVisualizer: Component = () => {
  const normalizedYs = createMemo(() => store.normalization.normalizedData);
  const originalYs = createMemo(() => store.trainingData?.ys || []);
  const method = createMemo(() => store.normalization.method);

  createEffect(() => {
    const plotDiv = document.getElementById('normalizationPlot');
    if (plotDiv && originalYs().length > 0 && normalizedYs()) {
      const data = [
        {
          x: originalYs(),
          y: normalizedYs()!,
          mode: 'markers',
          type: 'scatter',
          name: 'Normalized vs Original',
          marker: { color: colors.primary, size: 8 },
        }
      ];

      const layout = {
        title: 'Normalization Visualization',
        xaxis: { title: 'Original Y Values' },
        yaxis: { title: 'Normalized Y Values' },
        paper_bgcolor: colors.background,
        plot_bgcolor: colors.surface,
        font: { color: colors.text },
      };

      Plotly.newPlot(plotDiv, data, layout, { responsive: true });
    }
  });

  return (
    <div class={styles.container}>
      <h3 style={{ color: colors.text }}>Normalization Details</h3>
      <p style={{ color: colors.text }}>Method: {method()}</p>
      <div id="normalizationPlot" class={styles.plot}></div>
    </div>
  );
};

export default NormalizationVisualizer;