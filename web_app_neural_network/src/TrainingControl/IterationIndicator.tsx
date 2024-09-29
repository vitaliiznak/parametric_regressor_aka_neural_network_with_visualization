import { createEffect } from 'solid-js';

type IterationIndicatorProps = {
  iteration: number;
};

function IterationIndicator(props: IterationIndicatorProps) {
  createEffect(() => {
    // Trigger any side effects when the iteration changes
    console.log(`Starting iteration ${props.iteration}`);
  });

  return (
    <div class="iteration-indicator">
      <h2>Iteration {props.iteration}</h2>
    </div>
  );
}

export default IterationIndicator;