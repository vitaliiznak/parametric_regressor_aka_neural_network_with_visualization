import { Component, createSignal } from 'solid-js';
import NNVisualization from './NNVisualization';

export const App: Component = () => {
  const [dosage, setDosage] = createSignal(50);

  return (
    <div>
      <h1>Neural Network Visualization</h1>
      <NNVisualization dosage={dosage()} />
    </div>
  );
};