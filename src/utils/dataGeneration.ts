export interface DataPoint {
    x: number;
    y: number;
  }
  
  export function generateSampleData(count: number, addNoise: boolean = true): DataPoint[] {
    const data: DataPoint[] = [];
  
    for (let i = 0; i < count; i++) {
      const x = Math.random() * 100; // ChatGPT usage percentage (0-100)
      
      // Calculate the base y value
      let y = 50 * Math.sin(x / 10) + 30 * Math.exp(-Math.pow(x - 40, 2) / 1000) - x / 5 + 60;
      
      // Add random noise if enabled
      if (addNoise) {
        const noise = (Math.random() - 0.5) * 20; // Random value between -10 and 10
        y += noise;
      }
      
      // Clamp y between 0 and 100
      y = Math.max(0, Math.min(100, y));
      
      data.push({ x, y });
    }
  
    return data;
  }
  
  export function getTrueFunction(x: number): number {
    return 50 * Math.sin(x / 10) + 30 * Math.exp(-Math.pow(x - 40, 2) / 1000) - x / 5 + 60;
  }