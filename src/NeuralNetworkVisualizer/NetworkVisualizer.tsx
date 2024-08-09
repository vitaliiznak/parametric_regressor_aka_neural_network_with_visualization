import { Component, createEffect, onCleanup, onMount, createSignal } from "solid-js";
import { NetworkLayout } from "./layout";
import { NetworkRenderer } from "./renderer";
import { VisualNode, VisualNetworkData, VisualConnection } from "./types";
import { useAppStore } from "../AppContext";
import { debounce } from "@solid-primitives/scheduled";

interface NetworkVisualizerProps {
  includeLossNode: boolean;
  onVisualizationUpdate: () => void;
}

const NetworkVisualizer: Component<NetworkVisualizerProps> = (props) => {
  
  const [state, setState] = useAppStore();
  let canvasRef: HTMLCanvasElement | undefined;
  let containerRef: HTMLDivElement | undefined;
  let layoutCalculator: NetworkLayout | undefined;
  let renderer: NetworkRenderer | undefined;
  const [visualData, setVisualData] = createSignal<VisualNetworkData>({ nodes: [], connections: [] });
  let draggedNode: VisualNode | null = null;
  let isPanning = false;
  let lastPanPosition = { x: 0, y: 0 };

  const manageEventListeners = (action: 'add' | 'remove') => {
    const method = action === 'add' ? 'addEventListener' : 'removeEventListener';
    console.log(`manageEventListeners: ${action} listeners`);
    canvasRef?.[method]('mousedown', handleMouseDown as EventListener);
    canvasRef?.[method]('mousemove', handleMouseMove as EventListener);
    canvasRef?.[method]('mouseup', handleMouseUp as EventListener);
    canvasRef?.[method]('wheel', handleWheel as EventListener, { passive: false }); // Mark as non-passive
    canvasRef?.[method]('contextmenu', (e) => e.preventDefault());
  };

  const handleWheel = (e: WheelEvent) => {
    e.preventDefault();
    console.log("handleWheel called");
    if (renderer) {
      const rect = canvasRef!.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      renderer.zoom(x, y, delta);
      renderer.render(visualData());
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
    manageEventListeners('remove');
  });

  const initializeCanvas = () => {
    if (canvasRef && containerRef) {
      const { width, height } = containerRef.getBoundingClientRect();
      if (width > 0 && height > 0) {
        canvasRef.width = width;
        canvasRef.height = height;
        layoutCalculator = new NetworkLayout(canvasRef.width, canvasRef.height);
        renderer = new NetworkRenderer(canvasRef);
        updateVisualization();

        console.log('initializeCanvas', { width, height });

        manageEventListeners('add'); // Ensure event listeners are added here
      } else {
        console.warn('Container dimensions are zero, skipping canvas initialization');
      }
    }
  };
  
  const updateVisualization = debounce(() => {
    if (layoutCalculator && renderer) {
      const network = state.network;
      const networkData = network.toJSON();
      let newVisualData = layoutCalculator.calculateLayout(networkData);
  
      if (props.includeLossNode) {
        newVisualData = addLossFunctionNodes(newVisualData, network);
      }
      console.log('debhere', newVisualData);
      setVisualData(newVisualData);
      renderer.render(newVisualData);
      console.log('updateVisualization', { newVisualData });
      props.onVisualizationUpdate();
    }
  }, 100);

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
      weight: 1, // This could be updated with actual loss contribution if available
      bias: 1
    }));

    return {
      nodes: [...visualData.nodes, lossNode],
      connections: [...visualData.connections, ...lossConnections]
    };
  };

  const handleMouseDown = (e: MouseEvent) => {
    if (e.button === 2) { // Right mouse button
      isPanning = true;
      lastPanPosition = { x: e.clientX, y: e.clientY };
    } else if (canvasRef && layoutCalculator && renderer) {
      const rect = canvasRef.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      draggedNode = layoutCalculator.findNodeAt(
        x,
        y,
        visualData().nodes,
        renderer.scale,
        renderer.offsetX,
        renderer.offsetY
      );
    }
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (isPanning && renderer) {
      const dx = e.clientX - lastPanPosition.x;
      const dy = e.clientY - lastPanPosition.y;
      renderer.pan(dx, dy);
      lastPanPosition = { x: e.clientX, y: e.clientY };
      renderer.render(visualData());
    } else if (draggedNode && canvasRef && renderer) {
      const rect = canvasRef.getBoundingClientRect();
      const scaledX = (e.clientX - rect.left - renderer.offsetX) / renderer.scale;
      const scaledY = (e.clientY - rect.top - renderer.offsetY) / renderer.scale;
      draggedNode.x = scaledX;
      draggedNode.y = scaledY;
      renderer.render(visualData());
    }
  };

  const handleMouseUp = () => {
    isPanning = false;
    draggedNode = null;
  };

  createEffect(() => {
    console.log('Network data in NetworkVisualizer:', state.network);
    if (state.network && layoutCalculator) {
      const visualData = layoutCalculator.calculateLayout(state.network.toJSON());
      console.log('Calculated visual data:', visualData);
      setVisualData(visualData);
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