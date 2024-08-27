// src/components/CollapsibleSidebar.tsx
import { Component, createSignal, JSX } from "solid-js";
import { css } from "@emotion/css";
import { colors } from "../styles/colors";
import { FaSolidChevronLeft, FaSolidChevronRight } from 'solid-icons/fa';

const styles = {
  sidebar: css`
    background-color: ${colors.surface};
    transition: width 0.3s ease;
    overflow: hidden;
  `,
  toggleButton: css`
    position: absolute;
    top: 50%;
    left: 0;
    transform: translateY(-50%);
    background-color: ${colors.primary};
    color: ${colors.surface};
    border: none;
    padding: 8px;
    cursor: pointer;
    z-index: 10;
  `,
};

interface CollapsibleSidebarProps {
  children: JSX.Element;
}

const CollapsibleSidebar: Component<CollapsibleSidebarProps> = (props) => {
  const [isOpen, setIsOpen] = createSignal(true);

  return (
    <div
      class={css`
        ${styles.sidebar}
        width: ${isOpen() ? "300px" : "0px"};
      `}
    >
      <button
        class={styles.toggleButton}
        onClick={() => setIsOpen(!isOpen())}
        style={{ left: isOpen() ? "300px" : "0px" }}
      >
        {isOpen() ? <FaSolidChevronRight /> : <FaSolidChevronLeft />}
      </button>
      {isOpen() && props.children}
    </div>
  );
};

export default CollapsibleSidebar;