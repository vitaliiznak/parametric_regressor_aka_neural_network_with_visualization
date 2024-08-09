import { Component, createEffect, createSignal } from "solid-js";
import { useAppStore } from "../AppContext";
import { Trainer } from "../trainer";

const TrainingControls: Component<{ onVisualizationUpdate: () => void }> = (props) => {
  const [state, setState] = useAppStore();
  const [isRunning, setIsRunning] = createSignal(false);
  const [trainer, setTrainer] = createSignal<Trainer | null>(null);

  const startTraining = async () => {
    if (!state.trainingData?.xs || !state.trainingData?.ys) {
      console.error("Training data is not available");
      alert("Training data is not available");
      return;
    }
  
    setIsRunning(true);
    const newTrainer = new Trainer(state.network, state.trainingConfig);
    setTrainer(newTrainer);
  
    const trainingGenerator = newTrainer.train(state.trainingData.xs, state.trainingData.ys);
    for await (const result of trainingGenerator) {
      setState('trainingResult', result);
      setState('network', newTrainer.getNetwork()); // Update the network in the state
      props.onVisualizationUpdate();
      console.log('Training step completed, updating visualization');
      await new Promise(resolve => setTimeout(resolve, 0));
      if (!isRunning()) break;
    }
  
    setIsRunning(false);
  };

  const stopTraining = () => {
    setIsRunning(false);
  };

  const stepForward = () => {
    const currentTrainer = trainer();
    if (currentTrainer) {
      const result = currentTrainer.stepForward();
      console.log('Step Forward Result:', result);
      if (result) {
        setState('trainingResult', result);
        props.onVisualizationUpdate();
        console.log('State updated with new result');
      } else {
        console.log('No more steps available');
      }
    }
  };


  const stepBackward = () => {
    const currentTrainer = trainer();
    console.log('Step Backward');
    if (currentTrainer) {
      const result = currentTrainer.stepBackward();
      console.log('Step Backward Result:', result);
      if (result) {
        setState('trainingResult', result);
        props.onVisualizationUpdate();
      }
    }
  };

  // createEffect(() => {
  //   console.log("isRunning:", isRunning());
  //   console.log("trainer:", trainer());
  // });

  return (
    <div>
      <h3>Training Controls</h3>
      <button onClick={startTraining} disabled={isRunning()}>
        Start Training
      </button>
      <button onClick={stopTraining} disabled={!isRunning()}>
        Stop Training
      </button>
      <button onClick={stepForward} disabled={isRunning() || !trainer()}>
        Step Forward
      </button>
      <button onClick={stepBackward} disabled={isRunning() || !trainer()}>
        Step Backward
      </button>
    </div>
  );
};

export default TrainingControls;