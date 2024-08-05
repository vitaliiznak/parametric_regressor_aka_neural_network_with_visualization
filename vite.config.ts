import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';

export default defineConfig({
  plugins: [solidPlugin()],
  worker: {
    format: 'es',
  },
  server: {
    port: 3000,
  },
  root: './', // If your index.html is in the src folder
  publicDir: '../public', // Adjust this if your public assets are elsewhere
  build: {
       target: 'esnext',
    outDir: '../dist', // Specify where to output built files
    emptyOutDir: true,
  },
  resolve: {
    conditions: ['development', 'browser'],
  },
});