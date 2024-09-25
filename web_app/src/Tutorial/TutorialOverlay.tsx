// src/Tutorial/TutorialOverlay.tsx
import { Component, For } from "solid-js";
import { css } from "@emotion/css";
import { colors } from '../styles/colors';
import { typography } from '../styles/typography';

interface TutorialOverlayProps {
  step: number;
  onNext: () => void;
  onPrevious: () => void;
  onClose: () => void;
}

const tutorialSteps = [
  {
    title: "Welcome to Neural Network Basics",
    content: "In this tutorial, we'll explore the fundamentals of neural networks. Click 'Next' to begin!"
  },
  {
    title: "Neurons",
    content: "Neurons are the basic units of a neural network. They receive inputs, process them, and produce an output."
  },
  {
    title: "Layers",
    content: "Neural networks are organized in layers. We have input layers, hidden layers, and output layers."
  },
  // Add more steps as needed
];

const TutorialOverlay: Component<TutorialOverlayProps> = (props) => {
  return (
    <div class={styles.overlay}>
      <div class={styles.content}>
        <h2 class={styles.title}>{tutorialSteps[props.step].title}</h2>
        <p class={styles.text}>{tutorialSteps[props.step].content}</p>
        <div class={styles.navigation}>
          <button onClick={props.onPrevious} disabled={props.step === 0}>Previous</button>
          <button onClick={props.onNext} disabled={props.step === tutorialSteps.length - 1}>Next</button>
          <button onClick={props.onClose}>Close Tutorial</button>
        </div>
      </div>
    </div>
  );
};

const styles = {
  overlay: css`
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.7);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
  `,
  content: css`
    background-color: ${colors.surface};
    padding: 2rem;
    border-radius: 8px;
    max-width: 600px;
  `,
  title: css`
    font-size: ${typography.fontSize.xl};
    color: ${colors.primary};
    margin-bottom: 1rem;
  `,
  text: css`
    font-size: ${typography.fontSize.md};
    color: ${colors.text};
    margin-bottom: 1rem;
  `,
  navigation: css`
    display: flex;
    justify-content: space-between;
  `
};

export default TutorialOverlay;