import { Component } from 'solid-js';
import { css, keyframes } from '@emotion/css';
import { colors } from '../styles/colors';

const spin = keyframes`
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`;

const spinnerStyle = css`
  border: 4px solid ${colors.background};
  border-top: 4px solid ${colors.primary};
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: ${spin} 1s linear infinite;
`;

const Spinner: Component = () => {
  return <div class={spinnerStyle}></div>;
};

export default Spinner;