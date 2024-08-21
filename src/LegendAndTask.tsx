import { Component } from "solid-js";
import { css } from '@emotion/css';
import { FiCpu, FiTrendingUp } from 'solid-icons/fi';
import { colors } from './styles/colors';

const LegendAndTask: Component = () => {
  return (
    <div class={css`
      background-color: ${colors.background};
      border-radius: 16px;
      padding: 30px;
      margin-bottom: 40px;
      box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
      border: 1px solid ${colors.border};
      transition: all 0.3s ease;
      &:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
      }
    `}>
      <h2 class={css`
        color: ${colors.text};
        margin-bottom: 24px;
        font-size: 32px;
        font-weight: 700;
        text-align: center;
      `}>Neural Network Visualizer: The ChatGPT Productivity Paradox</h2>
      <p class={css`
        color: ${colors.textLight};
        line-height: 1.8;
        margin-bottom: 24px;
        font-size: 18px;
        text-align: center;
      `}>
        Explore the relationship between ChatGPT usage and developer productivity. 
        This interactive tool helps you find the optimal balance for maximum efficiency.
      </p>
      <div class={css`
        display: flex;
        justify-content: space-around;
        margin-top: 30px;
        background-color: ${colors.surface};
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
      `}>
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