import { Component, ErrorBoundary as SolidErrorBoundary, JSX } from 'solid-js';

interface ErrorBoundaryProps {
  fallback: (err: any, reset: () => void) => JSX.Element;
  children: JSX.Element;
}

const ErrorBoundary: Component<ErrorBoundaryProps> = (props) => {
  return (
    <SolidErrorBoundary fallback={props.fallback}>
      {props.children}
    </SolidErrorBoundary>
  );
};

export default ErrorBoundary;