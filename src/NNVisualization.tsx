/** @jsxImportSource solid-js */
import { Component, onMount, onCleanup } from 'solid-js';
import NeuralNetworkVisualizer, { CANVAS_HEIGHT, CANVAS_WIDTH, Connection, Node, NODE_HEIGHT, NODE_WIDTH } from './NeuralNetworkVisualizer/NeuralNetworkVisualizer';

const nodes: Node[] = [
  { id: 'input', label: 'Dosage', layer: 0, x: 0, y: 0, width: NODE_WIDTH, height: NODE_HEIGHT },
  { id: 'hidden1', label: 'Node 1', layer: 1, x: 0, y: 0, width: NODE_WIDTH, height: NODE_HEIGHT },
  { id: 'hidden2', label: 'Node 2', layer: 1, x: 0, y: 0, width: NODE_WIDTH, height: NODE_HEIGHT },
  { id: 'hidden3', label: 'Node 3', layer: 1, x: 0, y: 0, width: NODE_WIDTH, height: NODE_HEIGHT },
  { id: 'output1', label: 'Sum 1', layer: 2, x: 0, y: 0, width: NODE_WIDTH, height: NODE_HEIGHT },
  { id: 'output2', label: 'Sum 2', layer: 3, x: 0, y: 0, width: NODE_WIDTH, height: NODE_HEIGHT },
];

const connections: Connection[] = [
  { from: nodes[0], to: nodes[1], weight: -34.4, bias: 0.5 },
  { from: nodes[0], to: nodes[2], weight: -2.52, bias: -0.3 },
  { from: nodes[0], to: nodes[3], weight: 1.5, bias: 0.1 },
  { from: nodes[1], to: nodes[4], weight: -1.3, bias: 0.2 },
  { from: nodes[2], to: nodes[4], weight: 2.28, bias: -0.4 },
  { from: nodes[3], to: nodes[4], weight: 0.5, bias: 0.3 },
  { from: nodes[4], to: nodes[5], weight: 1.0, bias: -0.1 },
];

 const NNVisualization: Component = () => {
  let canvasRef: HTMLCanvasElement | undefined;
  let controller: NeuralNetworkVisualizer | undefined;

  onMount(() => {
    if (canvasRef) {
      controller = new NeuralNetworkVisualizer(canvasRef, nodes, connections);
    }
  });

  onCleanup(() => {
    // Clean up if necessary
  });

  const updateDosage = (e: Event) => {
    const value = (e.target as HTMLInputElement).value;
    controller?.updateDosage(Number(value));
  };

  return (
    <div>
      <canvas style={{border: '2px solid #aaaaaa', margin: '10px' }} ref={el => { canvasRef = el }} width={CANVAS_WIDTH} height={CANVAS_HEIGHT} />
      <input type="range" min="0" max="100" value={controller?.dosage} onInput={updateDosage} />
    </div>
  );
};

export default NNVisualization;