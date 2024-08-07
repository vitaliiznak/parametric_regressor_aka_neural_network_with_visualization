export class Value {
  #data: number;
  #grad: number;
  _prev: Set<Value>;
  _op: string;
  _backward: () => void;
  label: string;
  id: number; // Added a unique ID for each Value instance

  constructor(data: number, _children: Value[] = [], _op: string = '', label: string = '') {
    this.#data = data;
    this.#grad = 0;
    this._prev = new Set(_children);
    this._op = _op;
    this._backward = () => { };
    this.label = label;
    this.id = Value.idCounter++; // Assign a unique ID
  }

  static idCounter: number = 0; // Initialize the static idCounter property

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

  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();

    function buildTopo(v: Value) {
      if (!visited.has(v)) {
        visited.add(v);
        for (const child of v._prev) {
          buildTopo(child);
        }
        topo.push(v);
      }
    }

    buildTopo(this);
    this.grad = 1;
    for (const v of topo.reverse()) {
      v._backward();
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
}

Value.idCounter = 0; // Initialize the ID counter