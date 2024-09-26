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
  const data: DataPoint[] = [];
  for (let i = 0; i < count; i++) {
    const D = Math.random(); // Dosage amount from 0 to 100 mg
    const noise = addNoise ? (Math.random() - 0.5) * 0.05 : 0; // Noise between -0.025 and 0.025
    const I = getTrueFunction(D) + noise; // Immune response with optional noise
    data.push({ x: D, y: I });
  }
  return data;
}
/**
 * Computes the true function value based on the ChatGPT usage fraction.
 * 
 * @param x - ChatGPT usage as a fraction between 0 and 1.
 * @returns The computed productivity score as a fraction between 0 and 1.
 */
export function getTrueFunction(D: number): number {
  // Scale D from [0, 1] to [0, 100]
  const scaledD = D * 100;
  
  // Original immune response function with scaled D
  const response = (scaledD / 8) / ((scaledD / 8) ** 2 + 2) + 0.4 * Math.sin(0.45 * (scaledD / 8));
  
  // Normalize the result to keep it within the range (-1, 1)
  const I_min = -0.1; // Estimated minimum value of the function
  const I_max = 0.8;  // Estimated maximum value of the function
  
  // Normalization to scale the output between (-1, 1)
  const I_normalized = 2 * (response - I_min) / (I_max - I_min) - 1;
  
  return I_normalized;
}