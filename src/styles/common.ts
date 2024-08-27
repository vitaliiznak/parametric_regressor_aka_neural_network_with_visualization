import { css } from '@emotion/css';
import { colors } from './colors';
import { typography } from './typography';

export const spacing = {
  xs: '0.25rem',
  sm: '0.5rem',
  md: '1rem',
  lg: '1.5rem',
  xl: '2rem',
};

export const commonStyles = {
  button: css`
    font-family: ${typography.fontFamily};
    font-size: ${typography.fontSize.xs};
    font-weight: ${typography.fontWeight.medium};
    padding: 0.125rem 0.25rem;
    border-radius: 0.25rem;
    cursor: pointer;
    transition: background-color 0.2s, color 0.2s;
    
    color: ${colors.text};
    background-color: ${colors.surface};
    
    &:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
  `,
  
  primaryButton: css`
    background-color: ${colors.primary};
    color: ${colors.text};
    border: none;
    
    &:hover:not(:disabled) {
      background-color: ${colors.primaryDark};
    }
  `,
  
  secondaryButton: css`
    background-color: ${colors.secondary};
    color: ${colors.text};
    border: none;
    
    &:hover:not(:disabled) {
      background-color: ${colors.secondaryDark};
    }
  `,
  
  input: css`
    font-family: ${typography.fontFamily};
    font-size: ${typography.fontSize.xs};
    padding: 0.125rem 0.25rem;
    border: 1px solid ${colors.border};
    border-radius: 0.25rem;
    background-color: ${colors.surface};
    color: ${colors.text};
    
    &:focus {
      outline: none;
      border-color: ${colors.primary};
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
    }
  `,
  
  card: css`
    background-color: ${colors.surface};
    border-radius: ${spacing.md};
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    padding: ${spacing.md};
  `,
  
  label: css`
    font-family: ${typography.fontFamily};
    font-size: ${typography.fontSize.xs};
    font-weight: ${typography.fontWeight.medium};
    color: ${colors.textLight};
    margin-bottom: 0.25rem;
    display: block;
  `,

  formGroup: css`
    margin-bottom: ${spacing.md};
  `,

  errorText: css`
    color: ${colors.error};
    font-size: ${typography.fontSize.xs};
    margin-top: ${spacing.xs};
  `,

  sectionTitle: css`
    font-size: ${typography.fontSize.lg};
    font-weight: ${typography.fontWeight.bold};
    margin-bottom: 1rem;
  `,

  // Add responsive breakpoints
  breakpoints: {
    sm: '@media (min-width: 640px)',
    md: '@media (min-width: 768px)',
    lg: '@media (min-width: 1024px)',
    xl: '@media (min-width: 1280px)',
  },

  // // Add a flex container for easy layout
  // flexContainer: css`
  //   display: flex;
  //   flex-direction: column;
  //   gap: ${spacing.md};

  //   ${breakpoints.md} {
  //     flex-direction: row;
  //   }
  // `,
};