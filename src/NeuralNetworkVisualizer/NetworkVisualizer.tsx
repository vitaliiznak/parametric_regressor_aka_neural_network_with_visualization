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
  let isPanning = false;
  let lastPanPosition = { x: 0, y: 0 };

  const initializeCanvas = () => {
    if (canvasRef && containerRef) {
      const { width, height } = containerRef.getBoundingClientRect();
      canvasRef.width = width;
      canvasRef.height = height;
      layoutCalculator = new NetworkLayout(canvasRef.width, canvasRef.height);
      renderer = new NetworkRenderer(canvasRef);
      updateVisualization();

      console.log('initializeCanvas', { width, height } )
  
      canvasRef.addEventListener('mousedown', handleMouseDown);
      canvasRef.addEventListener('mousemove', handleMouseMove);
      canvasRef.addEventListener('mouseup', handleMouseUp);
      canvasRef.addEventListener('wheel', handleWheel);
      canvasRef.addEventListener('contextmenu', (e) => e.preventDefault());
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

  const handleWheel = (e: WheelEvent) => {
    e.preventDefault();
    console.log("renderer", renderer); 
    if (renderer) {
      const rect = canvasRef!.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      renderer.zoom(x, y, delta);
      renderer.render(visualData());
    }
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
    <div ref={containerRef} style={{ width: '100%', height: '600px', overflow: 'hidden' }}>
      <canvas ref={el => { 
        canvasRef = el;
        initializeCanvas();
      }} style={{ width: '100%', height: '100%' }} />
    </div>
  );
};

export default NetworkVisualizer;