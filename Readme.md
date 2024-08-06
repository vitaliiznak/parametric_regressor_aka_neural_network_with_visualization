# Neural Network Visualization

## Overview

This project is a web-based application for visualizing and training neural networks. It is built using Solid.js and TypeScript, and it leverages Vite for development and build processes. The application allows users to configure neural networks, visualize their structure, and train them interactively.

## Features

- **Neural Network Configuration**: Users can define the structure and activation functions of the neural network.
- **Training Visualization**: Real-time visualization of the training process, including loss values and network updates.
- **Interactive Controls**: Start, stop, and monitor the training process with interactive controls.
- **Network Layout and Rendering**: Custom layout and rendering of the neural network using HTML5 Canvas.

## Project Structure

### Main Components

- **App**: The main application component that sets up the context and renders the primary UI components.
  
```1:57:src/App.tsx
import { Component, createEffect } from 'solid-js';
import { createStore } from "solid-js/store";
import { MLP } from './NeuralNetwork/mlp';
import NetworkVisualizer from './NeuralNetworkVisualizer/NetworkVisualizer';
import TrainingControls from './TrainingControl/TrainingControls';
import NetworkConfigForm from './TrainingControl/NetworkConfigForm';
import TrainingConfigForm from './TrainingControl/TrainingConfigForm';
import TrainingStatus from './TrainingControl/TrainingStatus';
import { AppProvider } from "./AppContext";
import { CONFIG } from './config';
import { AppState } from './store';

const INITIAL_NETWORK = CONFIG.INITIAL_NETWORK;
const INITIAL_TRAINING = CONFIG.INITIAL_TRAINING;

const App: Component = () => {
  const initialState: AppState = {
    network: new MLP(INITIAL_NETWORK),
    trainingConfig: INITIAL_TRAINING,
    visualData: { nodes: [], connections: [] },
    dotString: '',
    lossValue: 0
  };

  const [store, setStore] = createStore<AppState>(initialState);

  createEffect(() => {
    console.log("Current store state:", store);
  });

  return (
    <AppProvider store={[store, setStore]}>
      <div>
        <h1>Neural Network Visualizer</h1>
        <div style={{ display: 'flex' }}>
          <div style={{ flex: 2 }}>
            <NetworkVisualizer includeLossNode={true} />
          </div>
          <div style={{ flex: 1 }}>
            <NetworkConfigForm />
            <TrainingConfigForm />
            <TrainingControls />
            <TrainingStatus />
            <div>
              <h2>Current Network Configuration</h2>
              <p>Layers: {store.network.layers.map(layer => layer.neurons.length).join(', ')}</p>
              <p>Activations: {store.network.activations.join(', ')}</p>
              <p>Current Loss: {store.lossValue.toFixed(4)}</p>
            </div>
          </div>
        </div>
      </div>
    </AppProvider>
  );
};
```


- **TrainingControls**: Provides buttons to start and stop the training process.
  
```1:60:src/TrainingControl/TrainingControls.tsx
import { Component, createSignal } from 'solid-js';
import { Trainer } from '../trainer';
import { useAppStore } from '../AppContext';

const TrainingControls: Component = () => {
  const [state, setState] = useAppStore();
  const [isTraining, setIsTraining] = createSignal(false);
  let trainerRef: Trainer | undefined;

  const startTraining = async () => {
    setIsTraining(true);
    console.log("Training started");
    const { network, trainingConfig } = state;
    trainerRef = new Trainer(network, trainingConfig);

    const xs = [[0], [0.5], [1]];
    const yt = [0, 0.5, 1];

    let lastUpdateTime = Date.now();

    try {
      for await (const result of trainerRef.train(xs, yt)) {
        console.log("Training iteration:", result);
        
        const currentTime = Date.now();
        if (currentTime - lastUpdateTime > 100) {  // Update every 100ms
          setState({ trainingResult: result });
          lastUpdateTime = currentTime;
        }
        
        if (!isTraining()) {
          console.log("Training stopped by user");
          break;
        }
      }
    } catch (error) {
      console.error("Error during training:", error);
    } finally {
      setIsTraining(false);
      console.log("Training finished");
    }
  };

  const stopTraining = () => {
    setIsTraining(false);
  };

  return (
    <div>
      <button onClick={startTraining} disabled={isTraining()}>
        Start Training
      </button>
      <button onClick={stopTraining} disabled={!isTraining()}>
        Stop Training
      </button>
    </div>
  );
};

export default TrainingControls;
```


- **NetworkVisualizer**: Renders the neural network structure and updates it during training.
  
```1:174:src/NeuralNetworkVisualizer/NetworkVisualizer.tsx
import { Component, createEffect, onCleanup, onMount, createSignal } from "solid-js";
import { NetworkLayout } from "./layout";
import { NetworkRenderer } from "./renderer";
import { VisualNode, VisualNetworkData, VisualConnection } from "./types";
import { useAppStore } from "../AppContext";
import { debounce } from "@solid-primitives/scheduled";

interface NetworkVisualizerProps {
  includeLossNode: boolean;
}

const NetworkVisualizer
```


- **TrainingConfigForm**: Form to configure training parameters like learning rate, epochs, and batch size.
  
```1:53:src/TrainingControl/TrainingConfigForm.tsx
import { Component, createSignal } from "solid-js";
import { useAppStore } from "../AppContext";

const TrainingConfigForm: Component = () => {
  const [state, setState] = useAppStore();
  const [config, setConfig] = createSignal(state.trainingConfig);

  const handleSubmit = (e: Event) => {
    e.preventDefault();
    setState({ trainingConfig: config() });
    console.log("Training config updated:", config());
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label>
          Learning Rate:
          <input
            type="number"
            step="0.001"
            value={config().learningRate}
            onInput={(e) => setConfig({ ...config(), learningRate: Number(e.currentTarget.value) })}
          />
        </label>
      </div>
      <div>
        <label>
          Epochs:
          <input
            type="number"
            value={config().epochs}
            onInput={(e) => setConfig({ ...config(), epochs: Number(e.currentTarget.value) })}
          />
        </label>
      </div>
      <div>
        <label>
          Batch Size:
          <input
            type="number"
            value={config().batchSize}
            onInput={(e) => setConfig({ ...config(), batchSize: Number(e.currentTarget.value) })}
          />
        </label>
      </div>
      <button type="submit">Update Training Config</button>
    </form>
  );
};
```


- **NetworkConfigForm**: Form to configure the neural network layers and activation functions.
  
```1:91:src/TrainingControl/NetworkConfigForm.tsx
import { Component, createEffect, createSignal } from "solid-js";
import { ActivationFunction } from "../NeuralNetwork/types";
import { MLP } from "../NeuralNetwork/mlp";
import { useAppStore } from "../AppContext";

const NetworkConfigForm
```


- **TrainingStatus**: Displays the current status of the training process, including the current epoch and loss.
  
```1:27:src/TrainingControl/TrainingStatus.tsx
import { Component, createEffect, createSignal, onCleanup } from "solid-js";
import { useAppStore } from "../AppContext";

const TrainingStatus: Component = () => {
  const [state, setState] = useAppStore();
  const [status, setStatus] = createSignal(state.trainingResult);

  createEffect(() => {
    const unsubscribe = () => {
      console.log("TrainingStatus Store updated:", state);
      setStatus(state.trainingResult);
    };

    onCleanup(unsubscribe);
  });

  return (
    <div>
      <h3>Training Status</h3>
      <p>Epoch: {status()?.epoch || 0}</p>
      <p>Loss: {status()?.loss.toFixed(4) || 'N/A'}</p>
    </div>
  );
};


export default TrainingStatus
```


### Core Logic

- **MLP (Multi-Layer Perceptron)**: Defines the structure and forward pass of the neural network.
  
```1:48:src/NeuralNetwork/mlp.ts
import { ActivationFunction, NetworkData, MLPConfig } from './types';
import { Layer } from './layer';
import { Value } from './value';

export class MLP {
  layers: Layer[];
  activations: ActivationFunction[];

  constructor(config: MLPConfig) {
    const { inputSize, layers, activations } = config;
    const sizes = [inputSize, ...layers];
    this.activations = activations;
    this.layers = sizes.slice(1).map((s, i) => 
      new Layer(sizes[i], s, activations[i] || 'tanh')
    );
  }

  forward(x: (number | Value)[]): Value | Value[] {
    let out: Value[] = x.map(Value.from);
    for (const layer of this.layers) {
      out = layer.forward(out);
    }
    return out.length === 1 ? out[0] : out;
  }

  parameters(): Value[] {
    return this.layers.flatMap(layer => layer.parameters());
  }

  zeroGrad(): void {
    this.parameters().forEach(p => p.grad = 0); 
  }

  toJSON(): NetworkData {
    return {
      layers: this.layers.map((layer, layerIndex) => ({
        id: `layer_${layerIndex}`,
        activations: this.activations,
        neurons: layer.neurons.map((neuron, neuronIndex) => ({
          id: `neuron_${layerIndex}_${neuronIndex}`,
          weights: neuron.w.map(w => w.data),
          bias: neuron.b.data,
          activation: neuron.activation
        }))
      }))
    };
  }
}
```


- **Trainer**: Handles the training loop and updates the network parameters.
  
```1:64:src/trainer.ts
import { MLP } from "./NeuralNetwork/mlp";
import { Value } from "./NeuralNetwork/value";


export interface TrainingConfig {
  learningRate: number;
  epochs: number;
  batchSize: number;
}

export interface TrainingResult {
  epoch: number;
  loss: number;
}
```


- **Value**: Represents a value in the network with support for automatic differentiation.
  
```1:176:src/NeuralNetwork/value.ts
export class Value {
  #data: number;
  #grad: number;
  _prev: Set<Value>;
  _op: string;
  _backward: () => void;
  label: string;
  id: number; // Added a unique ID for each Value instance

  constructor(data: number, _children: Value[] = [], _op: string = '', label: string = '') {
    this.#data = data;
    this.#grad = 0;
    this._prev = new Set(_children);
    this._op = _op;
    this._backward = () => { };
    this.label = label;
    this.id = Value.idCounter++; // Assign a unique ID
  }

  static from(n: number | Value): Value {
    return n instanceof Value ? n : new Value(n);
  }

  get data(): number {
    return this.#data;
  }

  set data(value: number) {
    this.#data = value;
  }

  get grad(): number {
    return this.#grad;
  }

  set grad(value: number) {
    this.#grad = value;
  }

  add(other: number | Value): Value {
    const otherValue = Value.from(other);
    const out = new Value(this.data + otherValue.data, [this, otherValue], '+');

    out._backward = () => {
      this.grad += out.grad;
      otherValue.grad += out.grad;
    };

    return out;
  }

  mul(other: number | Value): Value {
    const otherValue = Value.from(other);
    const out = new Value(this.data * otherValue.data, [this, otherValue], '*');

    out._backward = () => {
      this.grad += otherValue.data * out.grad;
      otherValue.grad += this.data * out.grad;
    };

    return out;
  }
```


- **Layer and Neuron**: Define the layers and neurons in the network.
  
```1:21:src/NeuralNetwork/layer.ts
import { Neuron } from './neuron';
import { ActivationFunction } from './types';
import { Value } from './value';


export class Layer {
  neurons: Neuron[];

  constructor(nin: number, nout: number, activation: ActivationFunction = 'tanh') {
    this.neurons = Array(nout).fill(0).map(() => new Neuron(nin, activation));
  }

  forward(x: Value[]): Value[] {
    return this.neurons.map(neuron => neuron.forward(x));
  }

  parameters(): Value[] {
    return this.neurons.flatMap(neuron => neuron.parameters());
  }
}
```

  
```1:43:src/NeuralNetwork/neuron.ts
// neuron.ts
import { ActivationFunction, NeuronData } from './types';
import { Value } from './value';


export class Neuron {
  w: Value[];
  b: Value;
  activation: ActivationFunction;

  constructor(nin: number, activation: ActivationFunction = 'tanh') {
    this.w = Array(nin).fill(0).map(() => new Value(Math.random() * 2 - 1));
    this.b = new Value(0);
    this.activation = activation;
  }

  forward(x: Value[]): Value {
    const act = x.reduce((sum, xi, i) => sum.add(this.w[i].mul(xi)), this.b);
    return this.applyActivation(act);
  }

  applyActivation(x: Value): Value {
    switch (this.activation) {
      case 'tanh': return x.tanh();
      case 'relu': return x.relu();
      case 'sigmoid': return x.sigmoid();
      case 'linear': default: return x;
    }
  }

  parameters(): Value[] {
    return [...this.w, this.b];
  }

  toJSON(): NeuronData {
    return {
      id: Math.random().toString(36).substr(2, 9),
      weights: this.w.map(w => w.data),
      bias: this.b.data,
      activation: this.activation
    };
  }
}
```


### Configuration

- **Configuration File**: Contains initial configurations for the network and training parameters.
  
```1:22:src/config.ts
import { MLPConfig } from "./NeuralNetwork/types";

export const CONFIG = {
  INITIAL_NETWORK: {
    inputSize: 3,
    layers: [4, 4, 1],
    activations: ['tanh', 'tanh']
  } as MLPConfig,
  INITIAL_TRAINING: {
    learningRate: 0.01,
    epochs: 1000,
    batchSize: 1
  },
  VISUALIZATION: {
    width: 800,
    height: 600,
    nodeWidth: 60,
    nodeHeight: 40,
    layerSpacing: 200,
    nodeSpacing: 80
  }
};
```


### Utility

- **AppContext**: Provides a global state for the application using Solid.js context.
  
```1:21:src/AppContext.tsx
import { createContext, useContext, ParentProps } from "solid-js";
import { Store, SetStoreFunction } from "solid-js/store";
import { AppState } from "./store";

const AppContext = createContext<[Store<AppState>, SetStoreFunction<AppState>]>();

export function AppProvider(props: ParentProps<{ store: [Store<AppState>, SetStoreFunction<AppState>] }>) {
  return (
    <AppContext.Provider value={props.store}>
      {props.children}
    </AppContext.Provider>
  );
}

export function useAppStore() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error("useAppStore must be used within an AppProvider");
  }
  return context;
}
```


- **ErrorBoundary**: A component to catch and display errors in the application.
  
```1:16:src/ErrorBoundary.tsx
import { Component, ErrorBoundary as SolidErrorBoundary, JSX } from 'solid-js';

interface ErrorBoundaryProps {
  fallback: (err: any, reset: () => void) => JSX.Element;
  children: JSX.Element;
}

const ErrorBoundary: Component<ErrorBoundaryProps> = (props) => {
  return (
    <SolidErrorBoundary fallback={props.fallback}>
      {props.children}
    </SolidErrorBoundary>
  );
};

export default ErrorBoundary;
```


## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/neural-network-visualization.git
   cd neural-network-visualization
   ```

2. Install dependencies:
   ```sh
   npm install
   # or
   yarn install
   ```

### Running the Application

To start the development server:
```sh
npm run dev
# or
yarn dev
```

Open your browser and navigate to `http://localhost:3000` to see the application in action.

### Building for Production

To build the application for production:
```sh
npm run build
# or
yarn build
```

The built files will be in the `dist` directory.

### Linting

To run the linter:
```sh
npm run lint
# or
yarn lint
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Solid.js](https://solidjs.com/)
- [Vite](https://vitejs.dev/)

---

Feel free to customize this README to better fit your project's needs.