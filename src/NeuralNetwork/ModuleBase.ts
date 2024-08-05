import Value from "./Value";

export default class ModuleBase {
  zeroGrad(): void {
    this.parameters().forEach(p => p.grad = 0);
  }

  parameters(): Value[] {
    return [];
  }
}