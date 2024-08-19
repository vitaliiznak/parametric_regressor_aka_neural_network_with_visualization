import { Component, Show, For, createMemo } from "solid-js";
import { store } from "../store";

const LearningProcessVisualizer: Component = () => {


  const renderData = createMemo(() => {
    if (!store.trainingResult) return null;

    const { step, data } = store.trainingResult;
    switch (step) {
      case 'forward':
        return (
          <div>
            <h4>Forward Pass</h4>
            <p>Input: {JSON.stringify(data.input)}</p>
            <p>Output: {JSON.stringify(data.output)}</p>
          </div>
        );
      case 'loss':
        return <div>Loss: {data.loss?.toFixed(4)}</div>;
      case 'backward':
        return (
          <div>
            <h4>Backward Pass</h4>
            <p>Gradients: {data.gradients?.map(g => g.toFixed(4)).join(', ')}</p>
          </div>
        );
      case 'update':
        return (
          <div>
            <h4>Weight Update</h4>
            <For each={data.oldWeights}>
              {(oldWeight, index) => (
                <p>
                  Weight {index()}: {oldWeight.toFixed(4)} → {data.newWeights?.[index()].toFixed(4)}
                </p>
              )}
            </For>
            <p>Learning Rate: {data.learningRate}</p>
          </div>
        );
      case 'iteration':
        return <div>Iteration {data.iteration} completed, Loss: {data.loss?.toFixed(4)}</div>;
      default:
        return null;
    }
  });

  return (
    <div>
      <h3>Learning Process</h3>
      <Show when={store.trainingResult}>
        <div>Current Step: {store.trainingResult?.step}</div>
        {renderData()}
      </Show>
    </div>
  );
};

export default LearningProcessVisualizer;