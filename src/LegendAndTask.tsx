import { Component } from "solid-js";
import { css } from '@emotion/css';
import { FiCpu, FiTrendingUp } from 'solid-icons/fi';

const LegendAndTask: Component = () => {
  return (
    <div class={css`
      background-color: #f0f8ff;
      border-radius: 16px;
      padding: 30px;
      margin-bottom: 40px;
      box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
      border: 1px solid #d0e8ff;
      transition: all 0.3s ease;
      &:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
      }
    `}>
      <h2 class={css`
        color: #2c3e50;
        margin-bottom: 24px;
        font-size: 32px;
        font-weight: 700;
        text-align: center;
      `}>The ChatGPT Productivity Paradox</h2>
      <p class={css`
        color: #34495e;
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
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
      `}>
        <div class={css`text-align: center;`}>
          <FiCpu size={32} color="#3498db" />
          <h3 class={css`color: #3498db; margin: 12px 0; font-size: 24px;`}>Input</h3>
          <p class={css`font-size: 18px; margin-bottom: 8px;`}>ChatGPT Usage (%)</p>
          <span class={css`font-size: 36px; font-weight: 700; color: #2980b9;`}>0-100</span>
        </div>
        <div class={css`text-align: center;`}>
          <FiTrendingUp size={32} color="#e74c3c" />
          <h3 class={css`color: #e74c3c; margin: 12px 0; font-size: 24px;`}>Output</h3>
          <p class={css`font-size: 18px; margin-bottom: 8px;`}>Productivity Score</p>
          <span class={css`font-size: 36px; font-weight: 700; color: #c0392b;`}>0-100</span>
        </div>
      </div>
    </div>
  );
};

export default LegendAndTask;