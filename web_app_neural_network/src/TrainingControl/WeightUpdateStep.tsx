import { Component, For } from "solid-js";
import { css } from "@emotion/css";
import { FaSolidArrowRight, FaSolidArrowDown } from 'solid-icons/fa';
import { colors } from '../styles/colors';

const WeightUpdateStep: Component<{ oldWeights: number[], newWeights: number[] }> = (props) => {
  const styles = {
    container: css`
      display: flex;
      flex-direction: column;
      align-items: center;
      background-color: ${colors.surface};
      border: 1px solid ${colors.border};
      border-radius: 0.25rem;
      padding: 0.5rem;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
      transition: transform 0.2s, box-shadow 0.2s;
      &:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      }
    `,
    stepIcon: css`
      font-size: 1rem;
      color: ${colors.primary};
    `,
    stepLabel: css`
      font-size: 0.75rem;
      font-weight: bold;
      margin-top: 0.25rem;
      color: ${colors.text};
    `,
    weightList: css`
      width: 100%;
      margin-top: 0.25rem;
      font-size: 0.625rem;
      color: ${colors.textLight};
    `,
    weightItem: css`
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.125rem;
    `,
    weightValue: css`
      font-family: monospace;
      color: ${colors.text};
    `,
    arrow: css`
      color: ${colors.secondary};
      margin: 0 0.125rem;
    `,
  };

  return (
    <div class={styles.container}>
      <div class={styles.stepIcon}>
        <FaSolidArrowDown />
      </div>
      <div class={styles.stepLabel}>Update</div>
      <div class={styles.weightList}>
        <For each={props.oldWeights}>
          {(oldWeight, index) => (
            <div class={styles.weightItem}>
              <span class={styles.weightValue}>{oldWeight.toFixed(2)}</span>
              <FaSolidArrowRight class={styles.arrow} />
              <span class={styles.weightValue}>{props.newWeights[index()].toFixed(2)}</span>
            </div>
          )}
        </For>
      </div>
    </div>
  );
};

export default WeightUpdateStep;