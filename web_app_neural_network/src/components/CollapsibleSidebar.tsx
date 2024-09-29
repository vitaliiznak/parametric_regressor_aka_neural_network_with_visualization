// src/components/CollapsibleSidebar.tsx
import { Component, createSignal, JSX } from "solid-js";
import { css } from "@emotion/css";
import { colors } from "../styles/colors";
import { FaSolidChevronLeft, FaSolidChevronRight } from "solid-icons/fa";

const CollapsibleSidebar: Component<{ children: JSX.Element }> = (props) => {
  const [isOpen, setIsOpen] = createSignal(true);

  // Adjust the width to make the sidebar wider by default
  const sidebarWidthOpen = 480; // Width of the sidebar when open (in pixels)
  const toggleButtonOffset = sidebarWidthOpen - 50; // Position of the toggle button when sidebar is open

  const styles = {
    container: css`
      position: relative;
      display: flex;
      height: 100%;
    `,
    sidebar: css`
      background-color: ${colors.surface};
      transition: width 0.3s ease;
      overflow: hidden;
      min-width: 0;
      position: relative;
      height: 100%;
    `,
    toggleButton: css`
      position: absolute;
      top: 50%;
      transform: translateY(-50%);
      background-color: ${colors.primary};
      color: ${colors.surface};
      border: none;
      padding: 0.5rem;
      cursor: pointer;
      z-index: 10;
      border-radius: 50%;

      &:focus {
        outline: 2px solid ${colors.text};
      }
    `,
    content: css`
      flex-grow: 1;
    `,
  };

  return (
    <div class={styles.container}>
      <div
        class={styles.sidebar}
        style={{ width: isOpen() ? `${sidebarWidthOpen}px` : "0px" }}
      >
        {props.children}
      </div>
      <button
        class={styles.toggleButton}
        style={{ left: isOpen() ? `${toggleButtonOffset}px` : "0px" }}
        onClick={() => setIsOpen(!isOpen())}
        aria-label={isOpen() ? "Collapse Sidebar" : "Expand Sidebar"}
      >
        {isOpen() ? <FaSolidChevronLeft /> : <FaSolidChevronRight />}
      </button>
      <div class={styles.content}>
        {/* Main content goes here */}
      </div>
    </div>
  );
};

export default CollapsibleSidebar;