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