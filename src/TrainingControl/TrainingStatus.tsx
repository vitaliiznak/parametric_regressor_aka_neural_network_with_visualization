import { Component } from "solid-js";
import { useAppStore } from "../AppContext";

const TrainingStatus: Component = () => {
  const [state] = useAppStore();

  return (
    <div>
      <h3>Training Status</h3>
      <p>Epoch: {state.trainingResult?.data.epoch || 0}</p>
      <p>Loss: {state.trainingResult?.data.loss?.toFixed(4) || 'N/A'}</p>
    </div>
  );
};


export default TrainingStatus