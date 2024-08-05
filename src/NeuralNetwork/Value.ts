export default class Value {
  data: number;
  grad: number;
  private _backward: () => void;
  private _prev: Set<Value>;
  private _op: string;

  constructor(data: number, _children: Value[] = [], _op: string = '') {
    this.data = data;
    this.grad = 0;
    this._backward = () => {};
    this._prev = new Set(_children);
    this._op = _op;
  }

  add(other: Value | number): Value {
    other = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data + other.data, [this, other], '+');

    out._backward = () => {
      this.grad += out.grad;
      other.grad += out.grad;
    };

    return out;
  }

  mul(other: Value | number): Value {
    other = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data * other.data, [this, other], '*');

    out._backward = () => {
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    };

    return out;
  }

  pow(exp: number): Value {
    const out = new Value(this.data ** exp, [this], `**${exp}`);

    out._backward = () => {
      this.grad += (exp * this.data ** (exp - 1)) * out.grad;
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
    const sig = 1 / (1 + Math.exp(-this.data));
    const out = new Value(sig, [this], 'Sigmoid');

    out._backward = () => {
      this.grad += sig * (1 - sig) * out.grad;
    };

    return out;
  }

  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this], 'Tanh');

    out._backward = () => {
      this.grad += (1 - t ** 2) * out.grad;
    };

    return out;
  }

  leakyRelu(alpha: number = 0.01): Value {
    const out = new Value(this.data > 0 ? this.data : alpha * this.data, [this], 'LeakyReLU');

    out._backward = () => {
      this.grad += (this.data > 0 ? 1 : alpha) * out.grad;
    };

    return out;
  }

  backward(): void {
    const topo: Value[] = [];
    const visited: Set<Value> = new Set();

    const buildTopo = (v: Value) => {
      if (!visited.has(v)) {
        visited.add(v);
        v._prev.forEach(child => buildTopo(child));
        topo.push(v);
      }
    };

    buildTopo(this);

    this.grad = 1;
    topo.reverse().forEach(v => v._backward());
  }

  toString(): string {
    return `Value(data=${this.data}, grad=${this.grad})`;
  }
}