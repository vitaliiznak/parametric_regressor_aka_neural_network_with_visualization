export type NormalizationMethod = 'none' | 'min-max' | 'standard';

export interface Normalizer {
  mean: number;
  std: number;
  min: number;
  max: number;
}

/**
 * Computes the mean, standard deviation, min, and max of the input data.
 * @param data Array of numerical inputs.
 * @param method The normalization method to compute parameters for.
 * @returns Normalizer object containing mean, std, min, and max.
 */
export function computeNormalizer(data: number[], method: NormalizationMethod): Normalizer {
  const mean = data.reduce((acc, val) => acc + val, 0) / data.length;
  const variance = data.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / data.length;
  const std = Math.sqrt(variance);
  const min = Math.min(...data);
  const max = Math.max(...data);
  
  return { mean, std, min, max };
}

/**
 * Applies normalization to a single data point based on the specified method.
 * @param value The input value to normalize.
 * @param normalizer The Normalizer containing mean, std, min, and max.
 * @param method The normalization method to apply.
 * @returns Normalized value.
 */
export function normalizeData(value: number, normalizer: Normalizer, method: NormalizationMethod): number {
  switch (method) {
    case 'min-max':
      return (value - normalizer.min) / (normalizer.max - normalizer.min);
    case 'standard':
      return (value - normalizer.mean) / normalizer.std;
    case 'none':
    default:
      return value;
  }
}