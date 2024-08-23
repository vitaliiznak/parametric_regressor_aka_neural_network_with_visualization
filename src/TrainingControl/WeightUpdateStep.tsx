import { Component, For } from "solid-js";
import { css } from "@emotion/css";
import { FaSolidArrowRight, FaSolidArrowDown } from 'solid-icons/fa';

const WeightUpdateStep: Component<{ oldWeights: number[], newWeights: number[] }> = (props) => {
  const styles = {
    container: css`
      display: flex;
      flex-direction: column;
      align-items: center;
      background-color: #e6f7ff;
      border: 2px solid #91d5ff;
      border-radius: 0.25rem;
      padding: 0.75rem;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      transition: transform 0.2s, box-shadow 0.2s;
      &:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      }
    `,
    stepIcon: css`
      font-size: 1.25rem;
      color: #1890ff;
    `,
    stepLabel: css`
      font-size: 0.875rem;
      font-weight: bold;
      margin-top: 0.25rem;
    `,
    weightList: css`
      width: 100%;
      margin-top: 0.5rem;
    `,
    weightItem: css`
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.25rem;
    `,
    weightValue: css`
      font-family: monospace;
      font-size: 0.75rem;
    `,
    arrow: css`
      color: #52c41a;
      margin: 0 0.25rem;
    `,
  };

  return (
    <div class={styles.container}>
      <div class={styles.stepIcon}>
        <FaSolidArrowDown />
      </div>
      <div class={styles.stepLabel}>Weight Update</div>
      <div class={styles.weightList}>
        <For each={props.oldWeights}>
          {(oldWeight, index) => (
            <div class={styles.weightItem}>
              <span class={styles.weightValue}>{oldWeight.toFixed(4)}</span>
              <FaSolidArrowRight class={styles.arrow} />
              <span class={styles.weightValue}>{props.newWeights[index()].toFixed(4)}</span>
            </div>
          )}
        </For>
      </div>
    </div>
  );
};

export default WeightUpdateStep;