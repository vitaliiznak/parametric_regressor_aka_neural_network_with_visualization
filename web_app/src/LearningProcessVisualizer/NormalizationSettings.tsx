import { Component } from "solid-js";
import { actions, store } from "../store";
import { NormalizationMethod } from "../utils/dataNormalization";
import { css } from "@emotion/css";
import { colors } from '../styles/colors';
import { typography } from '../styles/typography';

const styles = {
  container: css`
    padding: 1rem;
    background-color: ${colors.surface};
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
  `,
  label: css`
    margin-right: 0.5rem;
    color: ${colors.text};
    font-size: ${typography.fontSize.base};
  `,
  select: css`
    padding: 0.5rem;
    border: 1px solid ${colors.border};
    border-radius: 4px;
    background-color: ${colors.background};
    color: ${colors.text};
    font-size: ${typography.fontSize.base};
  `,
};

const NormalizationSettings: Component = () => {
  const handleChange = (e: Event) => {
    const target = e.target as HTMLSelectElement;
    const selectedMethod = target.value as NormalizationMethod;
    actions.setNormalizationMethod(selectedMethod);
  };

  return (
    <div class={styles.container}>
      <label class={styles.label} for="normalization-select">Normalization Method:</label>
      <select
        id="normalization-select"
        class={styles.select}
        onChange={handleChange}
        value={store.normalization.method}
      >
        <option value="none">None</option>
        <option value="min-max">Min-Max Normalization</option>
        <option value="standard">Standardization</option>
      </select>
    </div>
  );
};

export default NormalizationSettings;