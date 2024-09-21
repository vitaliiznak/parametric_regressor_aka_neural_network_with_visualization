export interface Normalizer {
    mean: number;
    std: number;
  }
  
  /**
   * Computes the mean and standard deviation of the input data.
   * @param data Array of numerical inputs.
   * @returns Normalizer object containing mean and std.
   */
  export function computeNormalizer(data: number[]): Normalizer {
    const mean = data.reduce((acc, val) => acc + val, 0) / data.length;
    const variance =
      data.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / data.length;
    const std = Math.sqrt(variance);
    return { mean, std };
  }
  
  /**
   * Standardizes a single input value.
   * @param value The input value to standardize.
   * @param normalizer The Normalizer containing mean and std.
   * @returns Standardized value.
   */
  export function standardizeInput(value: number, normalizer: Normalizer): number {
    return (value - normalizer.mean) / normalizer.std;
  }