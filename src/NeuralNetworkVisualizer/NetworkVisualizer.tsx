import { Component, createEffect, onCleanup, createSignal } from "solid-js";
import { css } from '@emotion/css';
import { NetworkLayout } from "./layout";
import { NetworkRenderer } from "./renderer";
import NeuronInfoSidebar from "./NeuronInfoSidebar";
import { store } from "../store";
import { VisualNetworkData, VisualNode } from "../types";
import { colors } from '../styles/colors';

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
  onSidebarToggle: (isOpen: boolean) => void;
}

const NetworkVisualizer: Component<NetworkVisualizerProps> = (props) => {
  const [visualData, setVisualData] = createSignal<VisualNetworkData>({ nodes: [], connections: [] });
  const [layoutCalculator, setLayoutCalculator] = createSignal<NetworkLayout | null>(null);
  const [renderer, setRenderer] = createSignal<NetworkRenderer | null>(null);
  const [canvasRef, setCanvasRef] = createSignal<HTMLCanvasElement | null>(null);
  const [containerRef, setContainerRef] = createSignal<HTMLDivElement | null>(null);
  const [isCanvasInitialized, setIsCanvasInitialized] = createSignal(false);
  const [isPanning, setIsPanning] = createSignal(false);
  const [selectedNeuron, setSelectedNeuron] = createSignal<VisualNode | null>(null);
  const [customNodePositions, setCustomNodePositions] = createSignal<Record<string, { x: number, y: number }>>({});

  let draggedNode: VisualNode | null = null;
  let mouseDownTimer: number | null = null;
  let initialMousePosition: { x: number; y: number } | null = null;
  let lastPanPosition: { x: number; y: number } = { x: 0, y: 0 };
  let animationFrameId: number | undefined;

  const calculateVisualData = () => {
    const layoutCalculatorValue = layoutCalculator();
    if (!layoutCalculatorValue) {
      console.error('Layout calculator is not initialized');
      return { nodes: [], connections: [] };
    }

    const networkData = store.network.toJSON();
    const newVisualData = layoutCalculatorValue.calculateLayout(
      networkData,
      store.currentInput,
      store.simulationResult,
      customNodePositions()
    );
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
    rendererValue.render(newVisualData, selectedNeuron());
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
      rendererValue.render(visualData(), selectedNeuron());
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
    const simulationResult = store.simulationResult;

    if (network || currentInput || simulationResult) {
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
      const clickedNode = layoutCalculator()!.findNodeAt(
        x,
        y,
        visualData().nodes,
        renderer()!.scale,
        renderer()!.offsetX,
        renderer()!.offsetY
      );
      if (clickedNode) {
        draggedNode = clickedNode;
        initialMousePosition = { x: e.clientX, y: e.clientY };
        mouseDownTimer = setTimeout(() => {
          mouseDownTimer = null;
        }, 200); // Set a 200ms timer to distinguish between click and drag
      } else {
        setIsPanning(true);
        lastPanPosition = { x: e.clientX, y: e.clientY };
      }
    }
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (mouseDownTimer !== null && initialMousePosition) {
      const dx = e.clientX - initialMousePosition.x;
      const dy = e.clientY - initialMousePosition.y;
      if (Math.sqrt(dx * dx + dy * dy) > 5) { // 5px threshold to start dragging
        clearTimeout(mouseDownTimer);
        mouseDownTimer = null;
      }
    }
    const canvas = canvasRef();
    const rendererValue = renderer();
    const visualDataVal = visualData();
    if (isPanning() && rendererValue) {
      const dx = e.clientX - lastPanPosition.x;
      const dy = e.clientY - lastPanPosition.y;
      rendererValue.pan(dx, dy);
      lastPanPosition = { x: e.clientX, y: e.clientY };
      rendererValue.render(visualDataVal, selectedNeuron());
    } else if (draggedNode && canvas && rendererValue && e.buttons === 1) {
      const rect = canvas.getBoundingClientRect();
      const scaledX = (e.clientX - rect.left - rendererValue.offsetX) / rendererValue.scale;
      const scaledY = (e.clientY - rect.top - rendererValue.offsetY) / rendererValue.scale;
      draggedNode.x = scaledX;
      draggedNode.y = scaledY;
      
      // Save the custom position
      setCustomNodePositions(prev => ({
        ...prev,
        [draggedNode.id]: { x: scaledX, y: scaledY }
      }));

      setVisualData({
        ...visualDataVal,
        nodes: visualDataVal.nodes.map(node => node.id === draggedNode?.id ? draggedNode : node)
      });
      rendererValue.render(visualDataVal, selectedNeuron());
    } else if (canvas && layoutCalculator && rendererValue) {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const hoveredNode = layoutCalculator()!.findNodeAt(
        x,
        y,
        visualDataVal.nodes,
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
        rendererValue.render(visualDataVal, selectedNeuron());
      }
    }
  };

  const handleMouseUp = () => {
    const canvas = canvasRef();
    if (canvas) {
      if (mouseDownTimer !== null) {
        clearTimeout(mouseDownTimer);
        // If the timer hasn't been cleared, it's a click
        if (draggedNode) {
          setSelectedNeuron(draggedNode);
          props.onSidebarToggle(true);
        }
      }
      setIsPanning(false);
      draggedNode = null;
      mouseDownTimer = null;
      initialMousePosition = null;
      canvas.style.cursor = 'grab';
    }
  };



  const containerStyle = css`
    width: 100%;
    height: 0;
    padding-bottom: 75%; // 4:3 aspect ratio
    position: relative;
    min-height: 400px;
    overflow: hidden;
    border: 1px solid ${colors.border};
    background-color: ${colors.surface};
    
    @media (max-width: 768px) {
      padding-bottom: 100%; // 1:1 aspect ratio on smaller screens
    }
  `;

  const canvasStyle = css`
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
  `;

  return (
    <div ref={setContainerRef} class={containerStyle}>
      <canvas
        ref={el => {
          setCanvasRef(el);
          if (el) initializeCanvas(el);
        }}
        class={canvasStyle}
        onMouseDown={handleMouseDown}
      />
      <NeuronInfoSidebar
        neuron={selectedNeuron()}
        onClose={() => {
          console.log("Closing sidebar");
          setSelectedNeuron(null);
          props.onSidebarToggle(false);
          render(performance.now()); // Re-render to remove the highlight
        }}
      />
    </div>
  );
};

export default NetworkVisualizer;