import Layer from "./Layer";
import ModuleBase from "./ModuleBase";
import Value from "./Value";

export default class MLP extends ModuleBase {
    layers: Layer[];
  
    constructor(nin: number, nouts: number[], activations: string[]) {
      super();
      if (nouts.length !== activations.length) {
        throw new Error('The length of nouts and activations arrays must match.');
      }
      const sz = [nin, ...nouts];
      this.layers = sz.slice(0, -1).map((s, i) => new Layer(s, sz[i + 1], activations[i]));
    }
  
    forward(x: Value[]): Value | Value[] {
      let output: Value[] = x;
      this.layers.forEach(layer => {
        output = layer.forward(output);
      });
      return output.length === 1 ? output[0] : output;
    }
  
    parameters(): Value[] {
      return this.layers.flatMap(layer => layer.parameters());
    }
  
    toString(): string {
      return `MLP of [${this.layers.join(', ')}]`;
    }
  }