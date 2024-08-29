import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';

export default defineConfig({
  plugins: [solidPlugin({
    babel: {
      plugins: ['@emotion/babel-plugin']
    }
  })],
  base: '/func_interpolator_aka_neural_network_with_vizualisation/',
  server: {
    port: 3000,
  },
  build: {
    target: 'esnext',
    outDir: 'dist', // Change this to 'dist'
    emptyOutDir: true,
  },
});