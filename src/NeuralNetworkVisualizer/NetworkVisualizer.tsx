import { Component, createEffect, onCleanup, createSignal } from "solid-js";
import { css } from '@emotion/css';
import { NetworkLayout } from "./layout";
import { NetworkRenderer } from "./renderer";
import { store } from "../store";
import { VisualNetworkData, VisualNode } from "../types";

// Create tooltip element
const tooltip = document.createElement('div');
tooltip.style.position = 'absolute';
tooltip.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
tooltip.style.color = 'white';
tooltip.style.padding = '5px';
tooltip.style.borderRadius = '5px';
tooltip.style.pointerEvents = 'none';
tooltip.style.display = 'none';
document.body.appendChild(tooltip);

const showTooltip = (x: number, y: number, text: string) => {
  tooltip.style.left = `${x + 10}px`;
  tooltip.style.top = `${y + 10}px`;
  tooltip.innerText = text;
  tooltip.style.display = 'block';
};

const hideTooltip = () => {
  tooltip.style.display = 'none';
};

interface NetworkVisualizerProps {
  includeLossNode: boolean;
  onVisualizationUpdate: () => void;
}

const NetworkVisualizer: Component<NetworkVisualizerProps> = (props) => {
  const [visualData, setVisualData] = createSignal<VisualNetworkData>({ nodes: [], connections: [] });
  const [layoutCalculator, setLayoutCalculator] = createSignal<NetworkLayout | null>(null);
  const [renderer, setRenderer] = createSignal<NetworkRenderer | null>(null);
  const [canvasRef, setCanvasRef] = createSignal<HTMLCanvasElement | null>(null);
  const [containerRef, setContainerRef] = createSignal<HTMLDivElement | null>(null);
  const [isCanvasInitialized, setIsCanvasInitialized] = createSignal(false);
  const [isPanning, setIsPanning] = createSignal(false);

  let draggedNode: VisualNode | null = null;

  let lastPanPosition: { x: number; y: number } = { x: 0, y: 0 };
  let animationFrameId: number | undefined;

  const calculateVisualData = () => {
    const layoutCalculatorValue = layoutCalculator();
    if (!layoutCalculatorValue) {
      console.error('Layout calculator is not initialized');
      return { nodes: [], connections: [] };
    }

    const networkData = store.network.toJSON();
    const newVisualData = layoutCalculatorValue.calculateLayout(networkData, store.currentInput, store.simulationOutput);
    return newVisualData;
  };

  const render = (time: number) => {
    const rendererValue = renderer()
    const newVisualData = calculateVisualData();
    setVisualData(newVisualData);
    if (!rendererValue) {
      console.warn('Renderer is not initialized');
      return
    }
    rendererValue.render(newVisualData, time);

  };

  const setupEventListeners = () => {
    const canvas = canvasRef();
    if (canvas) {
      const listeners = {
        mousedown: handleMouseDown,
        mousemove: handleMouseMove,
        mouseup: handleMouseUp,
        wheel: handleWheel,
        contextmenu: (e: Event) => e.preventDefault()
      };

      Object.entries(listeners).forEach(([event, handler]) => {
        canvas.addEventListener(event, handler as EventListener, { passive: false });
      });

      onCleanup(() => {
        Object.entries(listeners).forEach(([event, handler]) => {
          canvas.removeEventListener(event, handler as EventListener);
        });
      });
    }
  };

  const handleWheel = (e: WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const canvas = canvasRef();
    const rendererValue = renderer()
    if (rendererValue && canvas) {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      rendererValue.zoom(x, y, delta);
      rendererValue.render(visualData());
    }
  };

  const initializeCanvas = (canvasArg: HTMLCanvasElement) => {
    const canvas = canvasArg || canvasRef()
    const container = containerRef()
    if (!container || !containerRef) {
      console.error('Canvas or container ref is undefined');
      return;
    }
    const { width, height } = container.getBoundingClientRect();

    console.log('initializeCanvas', { width, height })
    if (canvas && width > 0 && height > 0) {
      canvas.width = width;
      canvas.height = height;
      setLayoutCalculator(new NetworkLayout(canvas.width, canvas.height));
      setRenderer(new NetworkRenderer(canvas));
      props.onVisualizationUpdate();
      setIsCanvasInitialized(true);
    } else {
      console.warn('Container dimensions are zero, skipping canvas initialization');
    }

  };

  createEffect(() => {
    if (isCanvasInitialized()) {
      console.log('Canvas initialized, setting up event listeners');
      setupEventListeners();
    } else {
      console.log('Canvas not initialized');
    }
  });

  createEffect(() => {
    const network = store.network;
    const currentInput = store.currentInput;
    const simulationOutput = store.simulationOutput;

    if (network || currentInput || simulationOutput) {
      render(performance.now());
    }
  });


  createEffect(() => {
    const container = containerRef()
    const canvas = canvasRef()
    if (container && canvas) {
      const resizeObserver = new ResizeObserver(() => {
        const { width, height } = container.getBoundingClientRect();
        if (width > 0 && height > 0) {
          initializeCanvas(canvas);
        }
      });
      resizeObserver.observe(container);

      onCleanup(() => {
        resizeObserver.disconnect();
      });
    }
  });

  onCleanup(() => {
    if (animationFrameId !== undefined) {
      cancelAnimationFrame(animationFrameId);
    }
  });

  const addLossFunctionNodes = (visualData: VisualNetworkData, network: any): VisualNetworkData => {
    const outputLayer = network.layers[network.layers.length - 1];
    const lossNodeId = 'loss';
    const lossNode: VisualNode = {
      id: lossNodeId,
      label: 'Loss',
      layerId: 'loss_layer',
      x: (network.layers.length + 1) * layoutCalculator()!.layerSpacing,
      y: layoutCalculator()!.canvasHeight / 2,
      weights: [],
      bias: 0
    };

    const newNodes = [...visualData.nodes, lossNode];
    const newConnections = [...visualData.connections];

    outputLayer.neurons.forEach((_, index) => {
      newConnections.push({
        from: `neuron_${network.layers.length - 1}_${index}`,
        to: lossNodeId,
        weight: 1,
        bias: 1
      });
    });

    return { nodes: newNodes, connections: newConnections };
  };

  const handleMouseDown = (e: MouseEvent) => {
    const canvas = canvasRef();
    if (canvas && layoutCalculator && renderer) {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      draggedNode = layoutCalculator()!.findNodeAt(
        x,
        y,
        visualData().nodes,
        renderer()!.scale,
        renderer()!.offsetX,
        renderer()!.offsetY
      );
      if (!draggedNode) {
        setIsPanning(true);
        lastPanPosition = { x: e.clientX, y: e.clientY };
      }
    }
  };

  const handleMouseMove = (e: MouseEvent) => {
    const canvas = canvasRef()
    const rendererValue = renderer()
    const visaulDataVal = visualData()
    if (isPanning() && rendererValue) {
      const dx = e.clientX - lastPanPosition.x;
      const dy = e.clientY - lastPanPosition.y;
      rendererValue.pan(dx, dy);
      lastPanPosition = { x: e.clientX, y: e.clientY };
      rendererValue.render(visaulDataVal);
    } else if (draggedNode && canvas && rendererValue) {
      const rect = canvas.getBoundingClientRect();
      const scaledX = (e.clientX - rect.left - rendererValue.offsetX) / rendererValue.scale;
      const scaledY = (e.clientY - rect.top - rendererValue.offsetY) / rendererValue.scale;
      if (draggedNode) {
        if (draggedNode) {
          draggedNode.x = scaledX;
          draggedNode.y = scaledY;
          setVisualData({
            ...visaulDataVal, nodes: visaulDataVal.nodes
              .map(node => node.id === draggedNode?.id ? draggedNode : node)
              .filter(node => node !== null) as VisualNode[]
          });
          rendererValue.render(visaulDataVal);
        }
      }
    } else if (canvas && layoutCalculator && rendererValue) {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const hoveredNode = layoutCalculator()!.findNodeAt(
        x,
        y,
        visaulDataVal.nodes,
        rendererValue.scale,
        rendererValue.offsetX,
        rendererValue.offsetY
      );
      if (hoveredNode) {
        canvas.style.cursor = 'pointer';
        // Show tooltip
        showTooltip(e.clientX, e.clientY, `Node: ${hoveredNode.label}\nOutput: ${hoveredNode.outputValue}`);
      } else {
        canvas.style.cursor = 'grab';
        hideTooltip();
        rendererValue.render(visaulDataVal);
      }
    }
  };

  const handleMouseUp = () => {
    const canvas = canvasRef()
    if (canvas) {
      setIsPanning(false);
      draggedNode = null;
      canvas.style.cursor = 'grab';
    }
  };



  const containerStyle = css`
    width: 100%;
    height: 840px;
    min-width: 200px;
    min-height: 400px;
    overflow: hidden;
    border: 1px solid black;
  `;

  const canvasStyle = css`
    width: 100%;
    height: 100%;
  `;

  return (
    <div ref={setContainerRef} class={containerStyle}>
      <canvas ref={el => {
        setCanvasRef(el);
        if (el) initializeCanvas(el);
      }} class={canvasStyle} />
    </div>
  );
};

export default NetworkVisualizer;