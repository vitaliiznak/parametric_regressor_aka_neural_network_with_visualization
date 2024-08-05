import { Component, createEffect, onCleanup, onMount, createSignal } from "solid-js";
import { NetworkLayout } from "./layout";
import { NetworkRenderer } from "./renderer";
import { VisualNode, VisualNetworkData } from "./types";
import { useAppStore } from "../AppContext";

const NetworkVisualizer: Component = () => {
  const store = useAppStore();
  let canvasRef: HTMLCanvasElement | undefined;
  let layoutCalculator: NetworkLayout | undefined;
  let renderer: NetworkRenderer | undefined;
  const [visualData, setVisualData] = createSignal<VisualNetworkData>({ nodes: [], connections: [] });
  let draggedNode: VisualNode | null = null;
  let lastUpdateTime = 0;

  onMount(() => {
    if (canvasRef) {
      layoutCalculator = new NetworkLayout(canvasRef.width, canvasRef.height);
      renderer = new NetworkRenderer(canvasRef);
      updateVisualization();

      canvasRef.addEventListener('mousedown', handleMouseDown);
      canvasRef.addEventListener('mousemove', handleMouseMove);
      canvasRef.addEventListener('mouseup', handleMouseUp);
    }
  });

  const updateVisualization = () => {
    const currentTime = Date.now();
    if (currentTime - lastUpdateTime < 100) {  // Throttle to max 10 updates per second
      return;
    }
    lastUpdateTime = currentTime;

    if (layoutCalculator && renderer) {
      const network = store.getState().network;
      const networkData = network.toJSON();
      const newVisualData = layoutCalculator.calculateLayout(networkData);
      setVisualData(newVisualData);
      renderer.render(newVisualData);
    }
  };

  const handleMouseDown = (e: MouseEvent) => {
    if (canvasRef && layoutCalculator) {
      const rect = canvasRef.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      draggedNode = layoutCalculator.findNodeAt(x, y, visualData().nodes);
    }
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (draggedNode && canvasRef) {
      const rect = canvasRef.getBoundingClientRect();
      draggedNode.x = e.clientX - rect.left - layoutCalculator!.nodeWidth / 2;
      draggedNode.y = e.clientY - rect.top - layoutCalculator!.nodeHeight / 2;

      // Ensure the node stays within the canvas
      draggedNode.x = Math.max(0, Math.min(draggedNode.x, canvasRef.width - layoutCalculator!.nodeWidth));
      draggedNode.y = Math.max(0, Math.min(draggedNode.y, canvasRef.height - layoutCalculator!.nodeHeight));

      renderer!.render(visualData());
    }
  };

  const handleMouseUp = () => {
    draggedNode = null;
  };

  createEffect(() => {
    const unsubscribe = store.subscribe((state) => {
      console.log("NetworkVisualizer: Store updated", state);
      updateVisualization();
    });

    onCleanup(() => {
      unsubscribe();
      if (canvasRef) {
        canvasRef.removeEventListener('mousedown', handleMouseDown);
        canvasRef.removeEventListener('mousemove', handleMouseMove);
        canvasRef.removeEventListener('mouseup', handleMouseUp);
      }
    });
  });


  createEffect(() => {
    const network = store.getState().network;
    console.log("NetworkVisualizer: Network updated", network);
    updateVisualization();
  });


  return <canvas ref={el => { 
    canvasRef = el;
    if (canvasRef) {
      layoutCalculator = new NetworkLayout(canvasRef.width, canvasRef.height);
      renderer = new NetworkRenderer(canvasRef);
      updateVisualization();
    }
  }} width={800} height={600} />;
};

export default NetworkVisualizer;