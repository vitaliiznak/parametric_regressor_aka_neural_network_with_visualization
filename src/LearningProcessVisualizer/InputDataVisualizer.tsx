import { Component, onMount, createEffect } from "solid-js";
import { useAppStore } from "../AppContext";
import Chart from "chart.js/auto";

const InputDataVisualizer: Component = () => {
  const [state] = useAppStore();
  let chartRef: HTMLCanvasElement | undefined;
  let chart: Chart | undefined;

  const createChart = () => {
    if (chartRef && state.trainingData) {
      const ctx = chartRef.getContext('2d');
      if (ctx) {
        chart = new Chart(ctx, {
          type: 'scatter',
          data: {
            datasets: [{
              label: 'House Prices',
              data: state.trainingData.xs.map((x, i) => ({
                x: x[0], // Size
                y: x[1], // Bedrooms
                r: state.trainingData!.ys[i] / 1000 // Price (scaled down for better visualization)
              })),
              backgroundColor: 'rgba(75, 192, 192, 0.6)'
            }]
          },
          options: {
            scales: {
              x: {
                title: {
                  display: true,
                  text: 'Size (sq m)'
                }
              },
              y: {
                title: {
                  display: true,
                  text: 'Number of Bedrooms'
                }
              }
            },
            plugins: {
              tooltip: {
                callbacks: {
                  label: (context: any) => {
                    return `Size: ${context.parsed.x} sq m, Bedrooms: ${context.parsed.y}, Price: $${context.raw.r * 1000}`;
                  }
                }
              }
            }
          }
        });
      }
    }
  };

  onMount(() => {
    createChart();
  });

  createEffect(() => {
    if (state.trainingData) {
      if (chart) {
        chart.destroy();
      }
      createChart();
    }
  });

  return (
    <div>
      <h3>Input Data Visualization</h3>
      <canvas ref={chartRef} width="400" height="200"></canvas>
    </div>
  );
};

export default InputDataVisualizer;