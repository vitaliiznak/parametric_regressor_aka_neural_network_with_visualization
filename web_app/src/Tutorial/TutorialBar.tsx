import { Component, createSignal, Show } from "solid-js";
import { css } from "@emotion/css";
import { FaSolidChevronDown, FaSolidChevronUp } from 'solid-icons/fa';
import { colors } from "../styles/colors";
import { typography } from "../styles/typography";

const styles = {
  container: css`
    position: relative;
    width: 100%;
    transition: height 0.3s ease;
    z-index: 1000;
    flex-shrink: 0;
  `,
  bar: css`
    background-color: ${colors.surface};
    border-top: 1px solid ${colors.border};
    height: 100%;
    overflow-y: auto;
  `,
  content: css`
    padding: 20px;
    max-width: 800px;
    margin: 0 auto;
    font-family: ${typography.fontFamily};
    color: ${colors.text};
  `,
  toggleButton: css`
    position: absolute;
    top: -30px;
    left: 50%;
    transform: translateX(-50%);
    background-color: ${colors.primary};
    color: ${colors.surface};
    border: none;
    padding: 8px 16px;
    cursor: pointer;
    border-radius: 8px 8px 0 0;
    display: flex;
    align-items: center;
    gap: 8px;
  `,
  resizeHandle: css`
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 5px;
    cursor: ns-resize;
    background-color: ${colors.border};
  `,
  title: css`
    font-size: 24px;
    color: ${colors.primary};
    margin-bottom: 16px;
  `,
  subtitle: css`
    font-size: 18px;
    color: ${colors.secondary};
    margin-bottom: 12px;
  `,
  paragraph: css`
    margin-bottom: 12px;
    line-height: 1.6;
  `,
  highlight: css`
    background-color: ${colors.primary}22;
    padding: 2px 4px;
    border-radius: 4px;
  `,
};

const TutorialBar: Component = () => {
  const [isOpen, setIsOpen] = createSignal(true);
  const [height, setHeight] = createSignal(400);
  let startY: number;
  let startHeight: number;

  const handleMouseDown = (e: MouseEvent) => {
    startY = e.clientY;
    startHeight = height();
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  const handleMouseMove = (e: MouseEvent) => {
    const diff = startY - e.clientY;
    setHeight(Math.max(200, Math.min(window.innerHeight - 100, startHeight + diff)));
  };

  const handleMouseUp = () => {
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  };

  return (
    <div
      class={styles.container}
      style={{
        height: `${isOpen() ? height() : 30}px`
      }}
    >
      <div class={styles.resizeHandle} onMouseDown={handleMouseDown} />
      <button
        class={styles.toggleButton}
        onClick={() => setIsOpen(!isOpen())}
      >
        {isOpen() ? <FaSolidChevronDown /> : <FaSolidChevronUp />}
        Tutorial
      </button>
      <Show when={isOpen()}>
        <div class={styles.bar}>
          <div class={styles.content}>
            <h1 class={styles.title}>Task: Crack Dr. House's Dosage Dilemma with Neural Networks! 🧪</h1>
            <h2 class={styles.subtitle}>Welcome to the AI-Powered Medical Challenge!</h2>
            <p class={styles.paragraph}>
              Hey there, future AI superstar! 👋 Ready to step into the intriguing world of Dr. House and tackle a medical mystery? Your mission, should you choose to accept it (insert dramatic heartbeat sound 🎵), is to assist the brilliant but enigmatic Dr. House in optimizing medication dosages for maximum patient effectiveness with minimal side effects!
            </p>
            <p class={styles.paragraph}>
              Here's the scoop on your thrilling task:
            </p>
            <ul>
              <li><span class={styles.highlight}>Input:</span> Medication dosage level (0% - 100%).</li>
              <li><span class={styles.highlight}>Output:</span> Patient effectiveness score (0% - 100%).</li>
              <li><span class={styles.highlight}>The Twist:</span> The relationship between dosage and effectiveness is as wavy as a heartbeat monitor! 💓</li>
            </ul>
            <p class={styles.paragraph}>
              We've cooked up a funky formula that captures this wild relationship. Your job? Train a neural network to match it and uncover the sweet spot where patient effectiveness peaks with minimal side effects! 🚀
            </p>
            <p class={styles.paragraph}>
              No fancy libraries here – you're building this neural network from scratch. It's like assembling a LEGO Death Star without instructions, but way cooler because it's AI! 🧠✨
            </p>
            <p class={styles.paragraph}>
              By the time you're done, you'll not only be Dr. House's dosage hero but also a neural network ninja. So, flex those coding muscles and get ready to dive into the wonderful world of AI!
            </p>
            <p class={styles.paragraph}>
              Remember, no pressure... but patient health (and the fate of the universe) depends on you! May the code be with you! 😄🖖
            </p>
          </div>
        </div>
      </Show>
    </div>
  );
};

export default TutorialBar;