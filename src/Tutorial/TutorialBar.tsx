import { Component, createSignal, Show } from "solid-js";
import { css } from "@emotion/css";
import { FaSolidChevronDown, FaSolidChevronUp } from 'solid-icons/fa';
import { colors } from "../styles/colors";
import { typography } from "../styles/typography";

const styles = {
  container: css`
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    transition: transform 0.3s ease;
    z-index: 1000;
  `,
  bar: css`
    background-color: ${colors.surface};
    border-top: 1px solid ${colors.border};
    height: 400px;
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

  return (
    <div
      class={styles.container}
      style={{
        transform: `translateY(${isOpen() ? '0' : 'calc(100% - 40px)'})`
      }}
    >
      <button
        class={styles.toggleButton}
        onClick={() => setIsOpen(!isOpen())}
      >
        {isOpen() ? <FaSolidChevronDown /> : <FaSolidChevronUp />}
        Tutorial
      </button>
      <div class={styles.bar}>
        <div class={styles.content}>
          <h1 class={styles.title}>Task: Boost Alice's Productivity with Neural Networks!</h1>
          <h2 class={styles.subtitle}>Welcome to the AI-Powered Productivity Challenge!</h2>
          <p class={styles.paragraph}>
            Hey there, future AI maestro! 👋 Ready to embark on a neural network adventure? Your mission, should you choose to accept it (cue dramatic music 🎵), is to help Alice, our junior developer, master the art of ChatGPT usage for maximum productivity!
          </p>
          <p class={styles.paragraph}>
            Here's the lowdown on your exciting task:
          </p>
          <ul>
            <li><span class={styles.highlight}>Input:</span> Alice's ChatGPT usage time (0-100%)</li>
            <li><span class={styles.highlight}>Output:</span> Her productivity score (0-100)</li>
            <li><span class={styles.highlight}>The Twist:</span> The relationship between ChatGPT time and productivity is as unpredictable as a cat on a hot tin roof! 🐱🔥</li>
          </ul>
          <p class={styles.paragraph}>
            We've cooked up a funky formula that captures this wild relationship. Your job? Train a neural network to match it and uncover the sweet spot where Alice's productivity skyrockets! 🚀
          </p>
          <p class={styles.paragraph}>
            No fancy libraries here – you're building this neural network from scratch. It's like assembling a LEGO Death Star without instructions, but way cooler because it's AI! 🧠✨
          </p>
          <p class={styles.paragraph}>
            By the time you're done, you'll not only be Alice's productivity hero but also a neural network ninja. So, flex those coding muscles and get ready to dive into the wonderful world of AI!
          </p>
          <p class={styles.paragraph}>
            Remember, no pressure... but Alice's productivity (and the fate of the universe) depends on you! May the code be with you! 😄🖖
          </p>
        </div>
      </div>
    </div>
  );
};

export default TutorialBar;