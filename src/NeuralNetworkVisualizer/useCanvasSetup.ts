import { createSignal, createEffect, onCleanup } from "solid-js";
import { NetworkLayout } from "./layout";
import { NetworkRenderer } from "./renderer";

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
      setLayoutCalculator(new NetworkLayout(canvas.width, canvas.height));
      setRenderer(new NetworkRenderer(canvas));
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