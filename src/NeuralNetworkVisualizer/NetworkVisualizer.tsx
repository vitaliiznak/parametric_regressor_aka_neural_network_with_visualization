import { Component, createEffect, onCleanup, createSignal, createMemo, Show, batch } from "solid-js";
import { NetworkRenderer } from "./renderer";
import NeuronInfoSidebar from "./NeuronInfoSidebar";
import { store } from "../store";
import { VisualNode } from "../types";
import { useCanvasSetup } from "./useCanvasSetup";
import { canvasStyle, containerStyle, tooltipStyle } from "./NetworkVisualizerStyles";
import { debounce } from "@solid-primitives/scheduled";
import { css } from "@emotion/css";
import { colors } from "../styles/colors";


interface NetworkVisualizerProps {
  includeLossNode: boolean;
  onVisualizationUpdate: () => void;
  onSidebarToggle: (isOpen: boolean) => void;
  onResize: (width: number, height: number) => void;
}

const NetworkVisualizer: Component<NetworkVisualizerProps> = (props) => {
  const {
    layoutCalculator,
    renderer,
    canvasRef,
    setCanvasRef,
    setContainerRef,
    isCanvasInitialized,
    initializeCanvas,
    handleResize
  } = useCanvasSetup(props.onVisualizationUpdate);

  const [isPanning, setIsPanning] = createSignal(false);
  const [selectedNeuron, setSelectedNeuron] = createSignal<VisualNode | null>(null);
  const [customNodePositions, setCustomNodePositions] = createSignal<Record<string, { x: number, y: number }>>({});
  const [tooltipData, setTooltipData] = createSignal<{ x: number, y: number, text: string } | null>(null);
  const [tutorialStep, setTutorialStep] = createSignal(0);

  let draggedNode: VisualNode | null = null;
  let mouseDownTimer: number | null = null;
  let lastPanPosition: { x: number; y: number } = { x: 0, y: 0 };

  const visualData = createMemo(() => {
    const layoutCalculatorValue = layoutCalculator();
    if (!layoutCalculatorValue) {
      console.warn('Layout calculator is not initialized');
      return { nodes: [], connections: [] };
    }

    const networkData = store.network.toJSON();
    return layoutCalculatorValue.calculateLayout(
      networkData,
      store.currentInput,
      store.simulationResult,
      customNodePositions()
    );
  });

  const currentSelectedNeuron = createMemo(() => {
    if (!selectedNeuron()) return null;
    return visualData().nodes.find(node => node.id === selectedNeuron()?.id) || null;
  });

  createEffect(() => {
    if (isCanvasInitialized()) {
      setupEventListeners();
    }
  });

  createEffect(() => {
    const rendererValue = renderer();
    const selected = selectedNeuron();
    if (rendererValue) {
      rendererValue.render(visualData(), selected);
    }
  });

  createEffect(() => {
    const rendererValue = renderer();
    if (rendererValue) {
      switch (tutorialStep()) {
        case 1: // Neurons step
          rendererValue.setHighlightedNode('neuron_0_0'); // Highlight the first neuron
          break;
        case 2: // Layers step
          rendererValue.setHighlightedNode(null); // Remove highlight
          break;
        // Add more cases for other steps
      }
    }
  });

  const setupEventListeners = () => {
    const canvas = canvasRef();
    if (canvas) {
      const listeners = {
        mousedown: handleMouseDown,
        mousemove: debouncedHandleMouseMove,
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

  const debouncedHandleMouseMove = debounce((e: MouseEvent) => {
    const canvas = canvasRef();
    const rendererValue = renderer();
    if (!canvas || !rendererValue) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (isPanning()) {
      handlePanning(e, rendererValue);
    } else if (draggedNode && e.buttons === 1) {
      handleNodeDragging(e, rendererValue);
    } else {
      handleHovering(x, y, rendererValue);
    }
  }, 4); // Debounce to roughly 60fps

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

        mouseDownTimer = setTimeout(() => {
          mouseDownTimer = null;
        }, 200); // Set a 200ms timer to distinguish between click and drag
      } else {
        setIsPanning(true);
        lastPanPosition = { x: e.clientX, y: e.clientY };
      }
    }
  };

  const handlePanning = (e: MouseEvent, rendererValue: NetworkRenderer) => {
    const dx = e.clientX - lastPanPosition.x;
    const dy = e.clientY - lastPanPosition.y;
    rendererValue.pan(dx, dy);
    lastPanPosition = { x: e.clientX, y: e.clientY };
    rendererValue.render(visualData(), selectedNeuron());
  };

  const handleNodeDragging = (e: MouseEvent, rendererValue: NetworkRenderer) => {
    const canvas = canvasRef();
    if (!canvas || !draggedNode) return;

    const rect = canvas.getBoundingClientRect();
    const scaledX = (e.clientX - rect.left - rendererValue.offsetX) / rendererValue.scale;
    const scaledY = (e.clientY - rect.top - rendererValue.offsetY) / rendererValue.scale;

    updateCustomNodePosition(draggedNode.id, scaledX, scaledY);
    rendererValue.render(visualData(), selectedNeuron());
  };

  const updateCustomNodePosition = (nodeId: string, x: number, y: number) => {
    setCustomNodePositions(prev => ({
      ...prev,
      [nodeId]: { x, y }
    }));
  };

  const showTooltip = (x: number, y: number, text: string) => {
    setTooltipData({ x, y, text });
  };

  const hideTooltip = () => {
    setTooltipData(null);
  };

  const handleHovering = (x: number, y: number, rendererValue: NetworkRenderer) => {
    const layoutCalculatorValue = layoutCalculator();
    if (!layoutCalculatorValue) return;

    const hoveredNode = layoutCalculatorValue.findNodeAt(
      x,
      y,
      visualData().nodes,
      rendererValue.scale,
      rendererValue.offsetX,
      rendererValue.offsetY
    );

    if (hoveredNode) {
      canvasRef()!.style.cursor = 'pointer';
      showTooltip(x, y, `Node: ${hoveredNode.label}\nOutput: ${hoveredNode.outputValue?.toFixed(4) || 'N/A'}`);
    } else {
      canvasRef()!.style.cursor = 'grab';
      hideTooltip();
    }
  };

  const handleMouseUp = () => {
    const canvas = canvasRef();
    if (canvas) {
      batch(() => {
        if (mouseDownTimer !== null) {
          clearTimeout(mouseDownTimer);
          if (draggedNode) {
            setSelectedNeuron(draggedNode);
            renderer()?.render(visualData(), draggedNode); // Trigger re-render with selected node
          }
        }
        setIsPanning(false);
        draggedNode = null;
        mouseDownTimer = null;
        lastPanPosition = { x: 0, y: 0 };
      });
      canvas.style.cursor = 'grab';
    }
  };

  return (
    <div
      ref={setContainerRef}
      class={css`
        ${containerStyle}
        resize: both;
        overflow: hidden;
        min-height: 400px;
        min-width: 300px;
      `}
      onResize={(e) => {
        const target = e.target as HTMLDivElement;
        props.onResize(target.clientWidth, target.clientHeight);
        handleResize();
      }}
    >
      <canvas
        ref={el => {
          setCanvasRef(el);
          if (el) initializeCanvas(el);
        }}
        class={canvasStyle}
        onMouseDown={handleMouseDown}
      />
      <NeuronInfoSidebar
        neuron={currentSelectedNeuron()}
        onClose={() => {
          setSelectedNeuron(null);
          renderer()?.render(visualData(), null);
        }}
      />
      <Show when={tooltipData()}>
        {(tooltipAccessor) => {
          const data = tooltipAccessor();
          return (
            <div
              class={css`
                ${tooltipStyle}
                background-color: ${colors.surface};
                color: ${colors.text};
                border: 1px solid ${colors.border};
                box-shadow: 0 2px 4px ${colors.shadow};
              `}
              style={{
                left: `${data.x}px`,
                top: `${data.y}px`,
                display: 'block'
              }}
            >
              {data.text}
            </div>
          );
        }}
      </Show>
   
  
    </div>
  );
};


export default NetworkVisualizer;