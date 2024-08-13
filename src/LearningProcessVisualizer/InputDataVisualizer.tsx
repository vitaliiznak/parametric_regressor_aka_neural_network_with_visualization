import { Component, onMount, createSignal, createEffect } from "solid-js";
import { useAppStore } from "../AppContext";
import Chart from "chart.js/auto";

const InputDataVisualizer: Component = () => {
  const [state] = useAppStore();
  const [chartRef, setChartRef] = createSignal<HTMLCanvasElement | null>(null);
  const [chart, setChart] = createSignal<Chart | null>(null);

  const createChart = () => {
    const chartEl = chartRef();
    if (chartEl && state.trainingData) {
      const ctx = chartEl.getContext('2d');
      if (ctx) {
        // Destroy existing chart if it exists
        const chartValue = chart();
        if (chartValue) {
          chartValue.destroy();
        }
        const newChart = new Chart(ctx, {
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
            responsive: true,
            maintainAspectRatio: false,
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
        setChart(newChart);
      }
    }
  };

  onMount(() => {
    // Use a small delay to ensure the DOM is fully rendered
    setTimeout(() => {
      createChart();
    }, 0);
  });

  createEffect(() => {
    if (state.trainingData) {
      // Use a small delay here as well
      setTimeout(() => {
        createChart();
      }, 0);
    }
  });

  return (
    <div style={{ width: '100%', height: '400px' }}>
      <h3>Input Data Visualization</h3>
      <canvas ref={setChartRef}></canvas>
    </div>
  );
};

export default InputDataVisualizer;