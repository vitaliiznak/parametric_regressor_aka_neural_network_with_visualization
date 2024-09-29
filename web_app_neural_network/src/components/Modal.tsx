import { Component, JSX } from "solid-js";
import { css } from "@emotion/css";

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  content: JSX.Element;
}

const Modal: Component<ModalProps> = (props) => {
  const styles = {
    overlay: css`
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background-color: rgba(0, 0, 0, 0.5);
      display: flex;
      justify-content: center;
      align-items: center;
    `,
    modal: css`
      background-color: white;
      padding: 2rem;
      border-radius: 0.5rem;
      max-width: 80%;
      max-height: 80%;
      overflow-y: auto;
    `,
    closeButton: css`
      position: absolute;
      top: 1rem;
      right: 1rem;
      background: none;
      border: none;
      font-size: 1.5rem;
      cursor: pointer;
    `,
  };

  return (
    <Show when={props.isOpen}>
      <div class={styles.overlay} onClick={props.onClose}>
        <div class={styles.modal} onClick={(e) => e.stopPropagation()}>
          <button class={styles.closeButton} onClick={props.onClose}>
            &times;
          </button>
          {props.content}
        </div>
      </div>
    </Show>
  );
};

export default Modal;