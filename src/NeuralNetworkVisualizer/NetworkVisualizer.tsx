import { Component, createEffect, onCleanup, onMount, createSignal, createMemo } from "solid-js";
import { NetworkLayout } from "./layout";
import { NetworkRenderer } from "./renderer";
import { VisualNode, VisualNetworkData } from "./types";
import { useAppStore } from "../AppContext";


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

  const [state] = useAppStore();
  const [visualData, setVisualData] = createSignal<VisualNetworkData>({ nodes: [], connections: [] });
  const [layoutCalculator, setLayoutCalculator] = createSignal<NetworkLayout | undefined>();
  const [renderer, setRenderer] = createSignal<NetworkRenderer | undefined>();
  const [canvasRef, setCanvasRef] = createSignal<HTMLCanvasElement | undefined>();
  const [containerRef, setContainerRef] = createSignal<HTMLDivElement | undefined>();
  const [isCanvasInitialized, setIsCanvasInitialized] = createSignal(false);

  let draggedNode: VisualNode | null = null;
  let isPanning = false;
  let lastPanPosition = { x: 0, y: 0 };

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
        canvas.addEventListener(event, handler as EventListener);
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
    console.log("handleWheel called");
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
      setupEventListeners();
    }
  });

  createEffect(() => {
    const layoutCalculatorValue = layoutCalculator();
    const rendererValue = renderer();
    if (layoutCalculatorValue && rendererValue) {

      const network = state.network;
      const networkData = network.toJSON();

      let newVisualData = layoutCalculatorValue.calculateLayout(networkData, state.simulationOutput);
      console.log('state.simulationOutput', state.simulationOutput);
      if (state.currentInput) {
        const currentInput = state.currentInput;
        newVisualData.nodes.forEach((node, index) => {
          if (node.layerId === 'input' && currentInput[index] !== undefined) {
            node.outputValue = currentInput[index];
          }
        });
      }

      if (state.simulationOutput) {
        const { input, layerOutputs } = state.simulationOutput;

        newVisualData.nodes.forEach((node, _index) => {
          const [nodeType, layerIndexStr, nodeIndexStr] = node.id.split('_');
          const layerIndex = parseInt(layerIndexStr);
          const nodeIndex = parseInt(nodeIndexStr);
          if (nodeType === 'input' && input[nodeIndex] !== undefined) {
            node.outputValue = input[nodeIndex];
          } else if (nodeType === 'neuron') {
            if (layerOutputs[layerIndex] && layerOutputs[layerIndex][nodeIndex] !== undefined) {
              node.outputValue = layerOutputs[layerIndex][nodeIndex];
            }
          }
        });
      }
      if (props.includeLossNode) {
        newVisualData = addLossFunctionNodes(newVisualData, network);
      }
      console.log('Updating visualization with new data:', newVisualData);

      setVisualData(newVisualData);
      rendererValue.render(newVisualData);
      props.onVisualizationUpdate();
    }
  })

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
    const canvas = canvasRef()
    if (e.button === 2) { // Right mouse button
      isPanning = true;
      lastPanPosition = { x: e.clientX, y: e.clientY };
    } else if (canvas && layoutCalculator && renderer) {
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
    }
  };

  const handleMouseMove = (e: MouseEvent) => {
    const canvas = canvasRef()
    const rendererValue = renderer()
    if (isPanning && rendererValue) {
      const dx = e.clientX - lastPanPosition.x;
      const dy = e.clientY - lastPanPosition.y;
      rendererValue.pan(dx, dy);
      lastPanPosition = { x: e.clientX, y: e.clientY };
      rendererValue.render(visualData());
    } else if (draggedNode && canvas && rendererValue) {
      const rect = canvas.getBoundingClientRect();
      const scaledX = (e.clientX - rect.left - rendererValue.offsetX) / rendererValue.scale;
      const scaledY = (e.clientY - rect.top - rendererValue.offsetY) / rendererValue.scale;
      if (draggedNode) {
        if (draggedNode) {
          draggedNode.x = scaledX;
          draggedNode.y = scaledY;
          setVisualData({
            ...visualData(), nodes: visualData().nodes
              .map(node => node.id === draggedNode?.id ? draggedNode : node)
              .filter(node => node !== null) as VisualNode[]
          });
          renderer()!.render(visualData());
        }
      }
    } else if (canvas && layoutCalculator && rendererValue) {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const hoveredNode = layoutCalculator()!.findNodeAt(
        x,
        y,
        visualData().nodes,
        renderer()!.scale,
        renderer()!.offsetX,
        renderer()!.offsetY
      );
      if (hoveredNode) {
        canvas.style.cursor = 'pointer';
        // Show tooltip
        showTooltip(e.clientX, e.clientY, `Node: ${hoveredNode.label}\nOutput: ${hoveredNode.outputValue}`);
      } else {
        canvas.style.cursor = 'default';
        hideTooltip();
        rendererValue.render(visualData());
      }
    }
  };

  const handleMouseUp = () => {
    isPanning = false;
    draggedNode = null;
  };

  createEffect(() => {
    console.log('Network data in NetworkVisualizer:', state.network);
    const layoutCalculatorValue = layoutCalculator()
    if (state.network && layoutCalculatorValue) {
      const visualData = layoutCalculatorValue.calculateLayout(state.network.toJSON(), state.simulationOutput);
      console.log('Calculated visual data:', visualData);
      setVisualData(visualData);
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
  
      // Trigger an initial resize event
      resizeObserver.disconnect();
      resizeObserver.observe(container);
  
      onCleanup(() => {
        resizeObserver.disconnect();
      });
    }
  });

return (
  <div ref={setContainerRef} style={{ width: '100%', height: '840px', minHeight: '400px', overflow: 'hidden', border: '1px solid black' }}>
    <canvas ref={el => {
      setCanvasRef(el);
      if (el) initializeCanvas(el);
    }} style={{ width: '100%', height: '100%' }} />
  </div>
);
};

export default NetworkVisualizer;