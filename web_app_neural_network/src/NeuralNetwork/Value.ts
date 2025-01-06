export class Value {
  #data: number;
  #grad: number;
  _prev: Set<Value>;
  _op: string;
  _backward: () => void;
  label: string;
  id: number;

  constructor(data: number, _children: Value[] = [], _op: string = '', label: string = '') {
    this.#data = data;
    this.#grad = 0;
    this._prev = new Set(_children);
    this._op = _op;
    this._backward = () => { };
    this.label = label;
    this.id = Value.idCounter++;
  }


  static idCounter: number = 0;

  static from(n: number | Value): Value {
    return n instanceof Value ? n : new Value(n);
  }

  get data(): number {
    return this.#data;
  }

  set data(value: number) {
    this.#data = value;
  }

  get grad(): number {
    return this.#grad;
  }

  set grad(value: number) {
    this.#grad = value;
  }

  add(other: number | Value): Value {
    const otherValue = Value.from(other);
    const out = new Value(this.data + otherValue.data, [this, otherValue], '+');

    out._backward = () => {
      this.grad += out.grad;
      otherValue.grad += out.grad;
    };

    return out;
  }

  sub(other: number | Value): Value {
    const otherValue = Value.from(other);
    const out = this.add(otherValue.mul(-1));

    out._backward = () => {
      this.grad += out.grad;
      otherValue.grad -= out.grad;
    };

    return out;
  }
  div(other: number | Value): Value {
    const otherValue = Value.from(other);
    const out = this.mul(otherValue.pow(-1));

    out._backward = () => {
      this.grad += (1 / otherValue.data) * out.grad;
      otherValue.grad -= (this.data / Math.pow(otherValue.data, 2)) * out.grad;
    };

    return out;
  }

  mul(other: number | Value): Value {
    const otherValue = Value.from(other);
    const out = new Value(this.data * otherValue.data, [this, otherValue], '*');

    out._backward = () => {
      this.grad += otherValue.data * out.grad;
      otherValue.grad += this.data * out.grad;
    };

    return out;
  }

  pow(n: number): Value {
    const out = new Value(Math.pow(this.data, n), [this], `^${n}`);

    out._backward = () => {
      this.grad += n * Math.pow(this.data, n - 1) * out.grad;
    };

    return out;
  }

  exp(): Value {
    const out = new Value(Math.exp(this.data), [this], 'exp');

    out._backward = () => {
      this.grad += out.data * out.grad;
    };

    return out;
  }

  tanh(): Value {
    const x = this.data;
    const t = (Math.exp(2 * x) - 1) / (Math.exp(2 * x) + 1);
    const out = new Value(t, [this], 'tanh');

    out._backward = () => {
      this.grad += (1 - t * t) * out.grad;
    };

    return out;
  }

  relu(): Value {
    const out = new Value(this.data < 0 ? 0 : this.data, [this], 'ReLU');

    out._backward = () => {
      this.grad += (out.data > 0 ? 1 : 0) * out.grad;
    };

    return out;
  }

  sigmoid(): Value {
    const s = 1 / (1 + Math.exp(-this.data));
    const out = new Value(s, [this], 'sigmoid');

    out._backward = () => {
      this.grad += s * (1 - s) * out.grad;
    };

    return out;
  }

  leakyRelu(alpha: number = 0.01): Value {
    const out = new Value(this.data > 0 ? this.data : alpha * this.data, [this], 'leaky-relu', this.label);

    out._backward = () => {
      this.grad += (this.data > 0 ? 1 : alpha) * out.grad;
    };

    return out;
  }

  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();
    const stack: Value[] = [this];
  
    while (stack.length > 0) {
      const v = stack[stack.length - 1];
      if (visited.has(v)) {
        stack.pop();
        topo.push(v);
      } else {
        visited.add(v);
        for (const child of v._prev) {
          if (!visited.has(child)) {
            stack.push(child);
          }
        }
      }
    }
  
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
  }

  toString(): string {
    return `Value(data=${this.data}, grad=${this.grad})`;
  }

  toDot(): string {
    const nodes: string[] = [];
    const edges: string[] = [];
    const visited = new Set<Value>();

    const traverse = (node: Value): string => {
      if (visited.has(node)) {
        return `node_${node.id}`;
      }
      visited.add(node);

      const nodeId = `node_${node.id}`;
      nodes.push(`${nodeId} [label="${node.label} (${node.data.toFixed(4)})"];`);

      if (node._prev.size > 0) {
        node._prev.forEach((child: Value) => {
          const childId = traverse(child);
          edges.push(`${childId} -> ${nodeId} [label="${node._op}"];`);
        });
      }

      return nodeId;
    };

    traverse(this);

    return `digraph G {
        rankdir=BT;
        ${nodes.join('\n')}
        ${edges.join('\n')}
    }`;
  }

  /**
   * Recursively builds a string representation of the computation tree.
   * @param indent - The current indentation level.
   * @returns A string representing the computation tree.
   */
  printTree(indent: string = ''): string {
    let treeStr = `${indent}${this.label || 'Value'} (${this._op || 'Input'}): ${this.data}\n`;
    if (this._prev.size > 0) {
      this._prev.forEach(child => {
        treeStr += child.printTree(indent + '  ');
      });
    }
    return treeStr;
  }

  /**
   * Retrieves a node from the computation tree by its label.
   * @param targetLabel - The label of the node to retrieve.
   * @returns The Value node with the specified label or undefined if not found.
   */
  findByLabel(targetLabel: string): Value | undefined {
    if (this.label === targetLabel) {
      return this;
    }
    for (const child of this._prev) {
      const result = child.findByLabel(targetLabel);
      if (result) {
        return result;
      }
    }
    return undefined;
  }
}
Value.idCounter = 0;

