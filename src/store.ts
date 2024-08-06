import { MLP } from "./NeuralNetwork/mlp";
import { VisualNetworkData } from "./NeuralNetworkVisualizer/types";
import { TrainingConfig, TrainingResult } from "./trainer";

type Listener<T> = (state: T) => void;

export class Store<T> {
  private state: T;
  private listeners: Listener<T>[] = [];

  constructor(initialState: T) {
    this.state = initialState;
  }

  getState(): T {
    return this.state;
  }

  setState(newState: Partial<T>) {
    this.state = { ...this.state, ...newState };
    console.log("Store updated:", this.state);
    this.notifyListeners();
  }

  subscribe(listener: Listener<T>) {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  private notifyListeners() {
    this.listeners.forEach(listener => listener(this.state));
  }
}

export interface AppState {
  network: MLP;
  trainingConfig: TrainingConfig;
  trainingResult?: TrainingResult;
  visualData: VisualNetworkData;
  dotString: string;
  lossValue: number; // Added this line
}

export const createAppStore = (initialState: AppState) => new Store(initialState);