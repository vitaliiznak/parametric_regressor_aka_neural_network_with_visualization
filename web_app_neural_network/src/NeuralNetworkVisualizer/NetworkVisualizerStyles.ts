import { css } from '@emotion/css';
import { colors } from '../styles/colors';

export const containerStyle = css`
  width: 100%;
  height: 0;
  padding-bottom: 75%;
  position: relative;
  min-height: 400px;
  overflow: hidden;
  border: 1px solid ${colors.border};
  background-color: #1B213D;
  
  @media (max-width: 768px) {
    padding-bottom: 100%;
  }
`;

export const canvasStyle = css`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
`;

export const tooltipStyle = css`
  position: fixed;
  display: none;
  background-color: ${colors.surface};
  border: 1px solid ${colors.border};
  padding: 5px;
  border-radius: 4px;
  font-size: 12px;
  pointer-events: none;
  z-index: 1000;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;