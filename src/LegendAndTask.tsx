import { Component } from "solid-js";
import { css } from '@emotion/css';
import { FiCpu, FiTrendingUp } from 'solid-icons/fi';
import { colors } from './styles/colors';
import { typography } from './styles/typography';
import { commonStyles, spacing } from './styles/common';

const styles = {
  container: css`
    ${commonStyles.card}
    margin-bottom: ${spacing.xl};
    transition: all 0.3s ease;
    &:hover {
      transform: translateY(-5px);
      box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
    }
  `,
  title: css`
    font-size: ${typography.fontSize['2xl']};
    font-weight: ${typography.fontWeight.bold};
    color: ${colors.text};
    margin-bottom: ${spacing.md};
  `,
  description: css`
    font-size: ${typography.fontSize.base};
    color: ${colors.textLight};
    margin-bottom: ${spacing.lg};
  `,
  infoCardsContainer: css`
    display: flex;
    justify-content: space-around;
    margin-top: ${spacing.lg};
    background-color: ${colors.background};
    border-radius: ${spacing.md};
    padding: ${spacing.lg};
  `,
};

const LegendAndTask: Component = () => {
  return (
    <div class={styles.container}>
      <h2 class={styles.title}>ChatGPT Productivity Paradox</h2>
      <p class={styles.description}>
        Explore the relationship between ChatGPT usage and productivity scores.
      </p>
      <div class={styles.infoCardsContainer}>
        <InfoCard 
          icon={<FiCpu size={32} color={colors.primary} />} 
          title="Input" 
          description="ChatGPT Usage (%)" 
          value="0-100" 
        />
        <InfoCard 
          icon={<FiTrendingUp size={32} color={colors.secondary} />} 
          title="Output" 
          description="Productivity Score" 
          value="0-100" 
        />
      </div>
    </div>
  );
};

const InfoCard: Component<{ icon: JSX.Element, title: string, description: string, value: string }> = ({ icon, title, description, value }) => (
  <div class={css`text-align: center;`}>
    {icon}
    <h3 class={css`color: ${colors.primary}; margin: 12px 0; font-size: 24px;`}>{title}</h3>
    <p class={css`font-size: 18px; margin-bottom: 8px;`}>{description}</p>
    <span class={css`font-size: 36px; font-weight: 700; color: ${colors.secondary};`}>{value}</span>
  </div>
);

export default LegendAndTask;