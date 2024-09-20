import { Component, Show } from "solid-js";
import { css } from "@emotion/css";
import { colors } from "../styles/colors";
import { VisualConnection } from "../types";

interface ConnectionSidebarProps {
  connection: VisualConnection | null;
  onClose: () => void;
}

const ConnectionSidebar: Component<ConnectionSidebarProps> = (props) => {
  return (
    <Show when={props.connection}>
      {(connection) => (
        <div class={styles.sidebar}>
          <button class={styles.closeButton} onClick={props.onClose}>×</button>
          <h2 class={styles.title}>Connection Details</h2>

          <div class={styles.detail}>
            <strong>From:</strong> {connection().from}
          </div>
          <div class={styles.detail}>
            <strong>To:</strong> {connection().to}
          </div>
          <div class={styles.detail}>
            <strong>Weight:</strong> {connection().weight.toFixed(4)}
          </div>
          <div class={styles.detail}>
            <strong>Weight Gradient:</strong> {connection().weightGradient?.toFixed(4) || 'N/A'}
          </div>
          <div class={styles.detail}>
            <strong>Bias:</strong> {connection().bias.toFixed(4)}
          </div>
          <div class={styles.detail}>
            <strong>Bias Gradient:</strong> {connection().biasGradient?.toFixed(4)}
          </div>
       
        </div>
      )}
    </Show>
  );
};

const styles = {
  sidebar: css`
    position: fixed;
    right: 0;
    top: 0;
    width: 250px;
    height: 100%;
    background-color: ${colors.surface};
    border-left: 1px solid ${colors.border};
    padding: 1rem;
    box-shadow: -2px 0 5px rgba(0,0,0,0.1);
    overflow-y: auto;
    z-index: 1001;
  `,
  closeButton: css`
    position: absolute;
    top: 10px;
    right: 10px;
    background: none;
    border: none;
    font-size: 1.2rem;
    cursor: pointer;
    color: ${colors.text};
  `,
  title: css`
    color: ${colors.primary};
    font-size: 1rem;
    margin-bottom: 1rem;
  `,
  detail: css`
    margin-bottom: 0.5rem;
    font-size: 0.875rem;
    color: ${colors.textLight};
  `,
};

export default ConnectionSidebar;