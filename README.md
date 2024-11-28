 Neural Network Visualization App

An interactive web application that visualizes the inner workings of a neural network in real-time. Built with Solid.js and TypeScript.

## Project Overview

This repository contains two main components:

1. **web_app_neural_network/** - Main web application codebase
2. **python_playground/** - Neural network experiments and prototypes in Python

## Features

- Real-time neural network visualization
- Interactive node and connection inspection
- Function visualization
- Training data visualization
- Customizable network parameters
- Responsive design

## Prerequisites

- Node.js (v16 or higher)
- Python 3.8+ (for python_playground)
- pnpm

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/web_app_neural_network.git
cd web_app_neural_network
```



### Web Application Setup

```bash
cd web_app_neural_network
pnpm install
```

### Python Environment Setup
```bash
cd python_playground
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```




## Running the Projects

### Web Application

Development mode:

```bash
cd web_app_neural_network
pnpm dev
```

This will start the development server at `http://localhost:3000`

### Production Build

```bash
npm run build
npm run serve
```

## Project Structure

- `/src/NeuralNetwork/` - Core neural network implementation
- `/src/NeuralNetworkVisualizer/` - Visualization components
- `/src/FunctionVisualizer/` - Function plotting components
- `/src/styles/` - Global styles and theme

## Technologies Used

- [Solid.js](https://www.solidjs.com/) - Frontend framework
- [TypeScript](https://www.typescriptlang.org/) - Programming language
- [Plotly.js](https://plotly.com/javascript/) - Data visualization
- [Emotion](https://emotion.sh/) - CSS-in-JS styling

## License

ISC License