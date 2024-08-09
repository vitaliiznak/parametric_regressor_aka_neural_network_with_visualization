import { Component, createEffect, onCleanup, onMount, createSignal, createMemo } from "solid-js";
import { NetworkLayout } from "./layout";
import { NetworkRenderer } from "./renderer";
import { VisualNode, VisualNetworkData, VisualConnection } from "./types";
import { useAppStore } from "../AppContext";

interface NetworkVisualizerProps {
  includeLossNode: boolean;
  onVisualizationUpdate: () => void;
}

const NetworkVisualizer: Component<NetworkVisualizerProps> = (props) => {

  const [state] = useAppStore();
  const [visualData, setVisualData] = createSignal<VisualNetworkData>({ nodes: [], connections: [] });
  const [layoutCalculator, setLayoutCalculator] = createSignal<NetworkLayout | undefined>();
  const [renderer, setRenderer] = createSignal<NetworkRenderer | undefined>();

  let canvasRef: HTMLCanvasElement | undefined;
  let containerRef: HTMLDivElement | undefined;

  let draggedNode: VisualNode | null = null;
  let isPanning = false;
  let lastPanPosition = { x: 0, y: 0 };

  createEffect(() => {
    if (canvasRef) {
      const listeners = {
        mousedown: handleMouseDown,
        mousemove: handleMouseMove,
        mouseup: handleMouseUp,
        wheel: handleWheel,
        contextmenu: (e: Event) => e.preventDefault()
      };

      Object.entries(listeners).forEach(([event, handler]) => {
        canvasRef!.addEventListener(event, handler as EventListener);
      });

      onCleanup(() => {
        Object.entries(listeners).forEach(([event, handler]) => {
          canvasRef!.removeEventListener(event, handler as EventListener);
        });
      });
    }
  });

  const handleWheel = (e: WheelEvent) => {
    e.preventDefault();
    console.log("handleWheel called");
    if (renderer()) {
      const rect = canvasRef!.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      renderer()!.zoom(x, y, delta);
      renderer()!.render(visualData());
    }
  };

  onMount(() => {
    console.log("onMount called");
    initializeCanvas();
    window.addEventListener('resize', initializeCanvas);
  });

  onCleanup(() => {
    console.log("onCleanup called");
    window.removeEventListener('resize', initializeCanvas);
  });

  const manageEventListeners = (action: 'add' | 'remove') => {
    const listeners = {
      mousedown: handleMouseDown,
      mousemove: handleMouseMove,
      mouseup: handleMouseUp,
      wheel: handleWheel,
      contextmenu: (e: Event) => e.preventDefault()
    };

    Object.entries(listeners).forEach(([event, handler]) => {
      if (canvasRef) {
        if (action === 'add') {
          canvasRef.addEventListener(event, handler as EventListener);
        } else {
          canvasRef.removeEventListener(event, handler as EventListener);
        }
      }
    });
  };

  const initializeCanvas = () => {
    if (!canvasRef || !containerRef) {
      console.error('Canvas or container ref is undefined');
      return;
    }
    const { width, height } = containerRef.getBoundingClientRect();
    if(width === 0 || height === 0) {
      console.error('Container dimensions are zero, skipping canvas initialization');
      return;
    }
    if (width > 0 && height > 0) {
      canvasRef.width = width;
      canvasRef.height = height;
      setLayoutCalculator(new NetworkLayout(canvasRef.width, canvasRef.height));
      setRenderer(new NetworkRenderer(canvasRef));
      manageEventListeners('add');
      renderer()?.render(visualData());
      props.onVisualizationUpdate();
    } else {
      console.warn('Container dimensions are zero, skipping canvas initialization');
    }

  };

  createEffect(() => {
    const layoutCalculatorValue = layoutCalculator();
    const rendererValue = renderer();
    if (layoutCalculatorValue && rendererValue) {

      const network = state.network;
      const networkData = network.toJSON();

      let newVisualData = layoutCalculatorValue.calculateLayout(networkData, state.simulationOutput);

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
        newVisualData.nodes.forEach((node, index) => {
          const [nodeType, indexStr] = node.id.split('_');
          const nodeIndex = parseInt(indexStr);
          if (nodeType === 'input' && input[nodeIndex] !== undefined) {
            node.outputValue = input[nodeIndex];
          } else if (node.layerId.startsWith('layer_')) {
            console.log('here layerOutputs', layerOutputs[0]);
            const layerIndex = parseInt(node.layerId.split('_')[1]);
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
      y: layoutCalculator()!.canvasHeight / 2
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
    if (e.button === 2) { // Right mouse button
      isPanning = true;
      lastPanPosition = { x: e.clientX, y: e.clientY };
    } else if (canvasRef && layoutCalculator && renderer) {
      const rect = canvasRef.getBoundingClientRect();
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
    if (isPanning && renderer) {
      const dx = e.clientX - lastPanPosition.x;
      const dy = e.clientY - lastPanPosition.y;
      renderer()!.pan(dx, dy);
      lastPanPosition = { x: e.clientX, y: e.clientY };
      renderer()!.render(visualData());
    } else if (draggedNode && canvasRef && renderer) {
      const rect = canvasRef.getBoundingClientRect();
      const scaledX = (e.clientX - rect.left - renderer()!.offsetX) / renderer()!.scale;
      const scaledY = (e.clientY - rect.top - renderer()!.offsetY) / renderer()!.scale;
      draggedNode.x = scaledX;
      draggedNode.y = scaledY;
      renderer()!.render(visualData());
    }
  };

  const handleMouseUp = () => {
    isPanning = false;
    draggedNode = null;
  };

  createEffect(() => {
    console.log('Network data in NetworkVisualizer:', state.network);
    if (state.network && layoutCalculator()) {
      const visualData = layoutCalculator()!.calculateLayout(state.network.toJSON(), state.simulationOutput);
      console.log('Calculated visual data:', visualData);
      setVisualData(visualData);
    }
  });


  createEffect(() => {
    if (containerRef) {
      const resizeObserver = new ResizeObserver(() => {
        initializeCanvas();
      });
      resizeObserver.observe(containerRef);

      onCleanup(() => {
        resizeObserver.disconnect();
      });
    }
  });

  return (
    <div ref={containerRef} style={{ width: '100%', height: '600px', overflow: 'hidden', border: '1px solid black' }}>
      <canvas ref={el => {
        canvasRef = el;
        if (canvasRef) {
          console.log("Canvas ref set");
          initializeCanvas(); // Ensure canvas is initialized here
        }
      }} style={{ width: '100%', height: '100%' }} />
    </div>
  );
};

export default NetworkVisualizer;