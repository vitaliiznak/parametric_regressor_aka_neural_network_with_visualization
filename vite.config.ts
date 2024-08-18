import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';

export default defineConfig({
  plugins: [solidPlugin()],
  resolve: {
    // alias: {
    //   'solid-js/store': 'node_modules/solid-js/store/dist/store.cjs',
    //   conditions: ['development', 'browser'],
    // },
  },
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

});