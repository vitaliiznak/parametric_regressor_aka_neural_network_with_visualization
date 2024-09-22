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
    const x = Math.random(); // ChatGPT usage fraction (0-1)

    // Calculate the base y value using the true function
    let y = getTrueFunction(x);

    // Add random noise if enabled
    if (addNoise) {
      const noise = (Math.random() - 0.5) * 0.2; // Random noise between -0.1 and 0.1
      y += noise;
    }

    // Clamp y between 0 and 1
    y = Math.max(0, Math.min(1, y));

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
  const scaledX = x * 100; // Scale x to [0, 100] for the original computation

  const y =
    50 * Math.sin(scaledX / 10) +
    30 * Math.exp(-Math.pow(scaledX - 40, 2) / 1000) -
    scaledX / 500 +
    60;

  // Assuming the original y ranges approximately between 0 and 100,
  // scale it back to [0, 1] for consistency.
  return y / 100;
}