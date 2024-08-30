import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';

export default defineConfig({
  plugins: [solidPlugin({
    babel: {
      plugins: ['@emotion/babel-plugin']
    }
  })],
  base: '/func_interpolator_aka_neural_network_with_visualization/', // Use the correct spelling here
  build: {
    target: 'esnext',
    outDir: 'dist',
    emptyOutDir: true,
  },
});