import { Component, createEffect, onCleanup, onMount, createSignal } from "solid-js";
import { NetworkLayout } from "./layout";
import { NetworkRenderer } from "./renderer";
import { VisualNode, VisualNetworkData, VisualConnection } from "./types";
import { useAppStore } from "../AppContext";

interface NetworkVisualizerProps {
  includeLossNode: boolean;
}

const NetworkVisualizer: Component<NetworkVisualizerProps> = (props) => {
  let throttleTimeout: ReturnType<typeof setTimeout> | null = null;
  
  const store = useAppStore();
  let canvasRef: HTMLCanvasElement | undefined;
  let containerRef: HTMLDivElement | undefined;
  let layoutCalculator: NetworkLayout | undefined;
  let renderer: NetworkRenderer | undefined;
  const [visualData, setVisualData] = createSignal<VisualNetworkData>({ nodes: [], connections: [] });
  let draggedNode: VisualNode | null = null;
  let lastUpdateTime = 0;

  const initializeCanvas = () => {
    if (canvasRef && containerRef) {
      const { width, height } = containerRef.getBoundingClientRect();
      canvasRef.width = width * 2;  // Increase canvas size
      canvasRef.height = height * 2;
      layoutCalculator = new NetworkLayout(canvasRef.width, canvasRef.height);
      renderer = new NetworkRenderer(canvasRef);
      updateVisualization();
  
      canvasRef.addEventListener('mousedown', handleMouseDown);
      canvasRef.addEventListener('mousemove', handleMouseMove);
      canvasRef.addEventListener('mouseup', handleMouseUp);
    }
  };
  
  onMount(() => {
    initializeCanvas();
    window.addEventListener('resize', initializeCanvas);
  });

  onCleanup(() => {
    window.removeEventListener('resize', initializeCanvas);
  });

  const updateVisualization = () => {
    if (throttleTimeout) return;

    throttleTimeout = setTimeout(() => {
      throttleTimeout = null;
      const currentTime = Date.now();
      if (currentTime - lastUpdateTime < 100) {
        return;
      }
      lastUpdateTime = currentTime;

      if (layoutCalculator && renderer) {
        const network = store.getState().network;
        const networkData = network.toJSON();
        let newVisualData = layoutCalculator.calculateLayout(networkData);

        // Add loss function nodes if includeLossNode is true
        if (props.includeLossNode) {
          newVisualData = addLossFunctionNodes(newVisualData, network);
        }

        setVisualData(newVisualData);
        renderer.render(newVisualData);
      }
    }, 100);
  };

  const addLossFunctionNodes = (visualData: VisualNetworkData, network: any): VisualNetworkData => {
    const outputLayer = network.layers[network.layers.length - 1];
    const lossNodeId = 'loss';
    const lossNode: VisualNode = {
      id: lossNodeId,
      label: 'Loss',
      layerId: 'loss_layer',
      x: (network.layers.length + 1) * layoutCalculator!.layerSpacing,
      y: layoutCalculator!.canvasHeight / 2
    };

    const lossConnections: VisualConnection[] = outputLayer.neurons.map((_, index) => ({
      from: `neuron_${network.layers.length - 1}_${index}`,
      to: lossNodeId,
      weight: 1 // This could be updated with actual loss contribution if available
    }));

    return {
      nodes: [...visualData.nodes, lossNode],
      connections: [...visualData.connections, ...lossConnections]
    };
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
    if (draggedNode && canvasRef && layoutCalculator) {
      const rect = canvasRef.getBoundingClientRect();
      draggedNode.x = e.clientX - rect.left;
      draggedNode.y = e.clientY - rect.top;

      // Ensure the node stays within the canvas
      draggedNode.x = Math.max(0, Math.min(draggedNode.x, canvasRef.width));
      draggedNode.y = Math.max(0, Math.min(draggedNode.y, canvasRef.height));

      renderer!.render(visualData());
    }
  };

  const handleMouseUp = () => {
    draggedNode = null;
  };

  createEffect(() => {
    const unsubscribe = store.subscribe(() => {
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

  return (
    <div ref={containerRef} style={{ width: '100%', height: '600px', overflow: 'auto' }}>
      <canvas ref={el => { 
        canvasRef = el;
        initializeCanvas();
      }} style={{ width: '100%', height: '100%' }} />
    </div>
  );
};

export default NetworkVisualizer;