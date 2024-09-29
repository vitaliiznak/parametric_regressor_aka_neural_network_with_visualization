import { Component, JSX, createSignal } from 'solid-js';
import { css } from '@emotion/css';
import { colors } from '../styles/colors';

interface TooltipProps {
  content: string;
  children: JSX.Element;
}

const Tooltip: Component<TooltipProps> = (props) => {
  const [isVisible, setIsVisible] = createSignal(false);

  const styles = {
    container: css`
      position: relative;
      display: inline-block;
    `,
    tooltip: css`
      position: absolute;
      bottom: 100%;
      left: 50%;
      transform: translateX(-50%);
      background-color: ${colors.text};
      color: ${colors.surface};
      padding: 8px 12px;
      border-radius: 4px;
      font-size: 14px;
      white-space: nowrap;
      z-index: 1000;
      opacity: ${isVisible() ? 1 : 0};
      transition: opacity 0.2s;
      pointer-events: none;
    `,
  };

  return (
    <div
      class={styles.container}
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {props.children}
      <div class={styles.tooltip}>{props.content}</div>
    </div>
  );
};

export default Tooltip;