import { createSignal, createEffect, onCleanup } from "solid-js";
import { NetworkLayout } from "./layout";
import { NetworkRenderer } from "./renderer";
import { store } from '../store'; // Ensure this import is present

export function useCanvasSetup(onVisualizationUpdate: () => void) {
  const [layoutCalculator, setLayoutCalculator] = createSignal<NetworkLayout | null>(null);
  const [renderer, setRenderer] = createSignal<NetworkRenderer | null>(null);
  const [canvasRef, setCanvasRef] = createSignal<HTMLCanvasElement | null>(null);
  const [containerRef, setContainerRef] = createSignal<HTMLDivElement | null>(null);
  const [isCanvasInitialized, setIsCanvasInitialized] = createSignal(false);

  const initializeCanvas = (canvas: HTMLCanvasElement) => {
    const container = containerRef();
    if (!container) {
      console.error('Container ref is undefined');
      return;
    }
    const { width, height } = container.getBoundingClientRect();

    if (width > 0 && height > 0) {
      canvas.width = width;
      canvas.height = height;

      // Initialize the layout calculator
      const layout = new NetworkLayout(canvas.width, canvas.height);
      setLayoutCalculator(layout);

      // Get the visual data to compute the bounding box
      const networkData = store.network.toJSON();
      const visualData = layout.calculateLayout(
        networkData,
        store.currentInput,
        store.simulationResult
      );

      // Compute the bounding box of the network
      const nodes = visualData.nodes;
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      nodes.forEach(node => {
        minX = Math.min(minX, node.x);
        minY = Math.min(minY, node.y);
        maxX = Math.max(maxX, node.x + layout.nodeWidth);
        maxY = Math.max(maxY, node.y + layout.nodeHeight);
      });

      // Define a top padding value
      const topPadding = 150; 

      // Calculate initial offsets to position the network at the top center
      const networkWidth = maxX - minX;
      const initialOffsetX = (canvas.width - networkWidth) / 2 -
      100 - minX;
      const initialOffsetY = topPadding - minY;

      // Initialize the renderer
      const rendererInstance = new NetworkRenderer(canvas);
      rendererInstance.offsetX = initialOffsetX;
      rendererInstance.offsetY = initialOffsetY;
      setRenderer(rendererInstance);

      onVisualizationUpdate();
      setIsCanvasInitialized(true);
    } else {
      console.warn('Container dimensions are zero, retrying in 100ms');
      setTimeout(() => initializeCanvas(canvas), 100);
    }
  };

  const handleResize = () => {
    const container = containerRef();
    const canvas = canvasRef();
    if (container && canvas) {
      const { width, height } = container.getBoundingClientRect();
      if (width > 0 && height > 0) {
        canvas.width = width;
        canvas.height = height;
        layoutCalculator()?.updateDimensions(width, height);
        renderer()?.updateDimensions(width, height);
        onVisualizationUpdate();
      }
    }
  };

  createEffect(() => {
    const container = containerRef();
    const canvas = canvasRef();
    if (container && canvas) {
      const resizeObserver = new ResizeObserver(handleResize);
      resizeObserver.observe(container);

      onCleanup(() => {
        resizeObserver.disconnect();
      });
    }
  });

  return {
    layoutCalculator,
    renderer,
    canvasRef,
    setCanvasRef,
    setContainerRef,
    isCanvasInitialized,
    initializeCanvas,
    handleResize
  };
}