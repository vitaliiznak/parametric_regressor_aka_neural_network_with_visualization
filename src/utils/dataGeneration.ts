export interface DataPoint {
  x: number; // ChatGPT usage as a fraction between 0 and 1
  y: number;
}

/**
 * Generates sample data points for the neural network.
 * 
 * @param count - Number of data points to generate.
 * @param addNoise - Whether to add random noise to the data.
 * @returns An array of DataPoint objects.
 */
export function generateSampleData(count: number, addNoise: boolean = true): DataPoint[] {
  const data = [];
  for (let i = 0; i < count; i++) {
    const x = Math.random(); // Dosage from 0 to 1
    const noise = (Math.random() - 0.5) * 0.05; // Noise between -0.025 and 0.025
    const y = (Math.sin(2 * Math.PI * x) 
            + 0.5 * Math.sin(4 * Math.PI * x) 
            + 1 
            + noise) / 3;
    data.push({ x, y });
  }
  return data;
}

/**
 * Computes the true function value based on the ChatGPT usage fraction.
 * 
 * @param x - ChatGPT usage as a fraction between 0 and 1.
 * @returns The computed productivity score as a fraction between 0 and 1.
 */
export function getTrueFunction(x: number): number {
  return (Math.sin(2 * Math.PI * x) + 0.5 * Math.sin(4 * Math.PI * x) + 1) / 3;
}