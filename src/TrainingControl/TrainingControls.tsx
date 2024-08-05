import { Component, createSignal } from 'solid-js';
import { Trainer } from '../trainer';
import { useAppStore } from '../AppContext';

const TrainingControls: Component = () => {
  const store = useAppStore();
  const [isTraining, setIsTraining] = createSignal(false);
  let trainerRef: Trainer | undefined;

  const startTraining = async () => {
    setIsTraining(true);
    console.log("Training started");
    const { network, trainingConfig } = store.getState();
    trainerRef = new Trainer(network, trainingConfig);

    const xs = [[0], [0.5], [1]];
    const yt = [0, 0.5, 1];

    let lastUpdateTime = Date.now();

    try {
      for await (const result of trainerRef.train(xs, yt)) {
        console.log("Training iteration:", result);
        
        const currentTime = Date.now();
        if (currentTime - lastUpdateTime > 100) {  // Update every 100ms
          store.setState({ trainingResult: result });
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