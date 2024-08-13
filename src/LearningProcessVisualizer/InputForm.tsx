import { Component, createSignal } from "solid-js";
import { useAppStore } from "../AppContext";

const InputForm: Component = () => {
  const [state, setState] = useAppStore();
  const [size, setSize] = createSignal("");
  const [bedrooms, setBedrooms] = createSignal("");
  const [age, setAge] = createSignal("");

  const handleSubmit = (e: Event) => {
    e.preventDefault();
    const values = [Number(size()), Number(bedrooms()),/*  Number(age() )*/];
    if (values.some(isNaN)) {
      alert("Please provide valid numbers for all inputs");
      return;
    }
    setState('currentInput', values);
  };

  return (
    <form onSubmit={handleSubmit}>
      <label>
        Size (sq m):
        <input type="number" value={size()} onInput={(e) => setSize(e.currentTarget.value)} />
      </label>
      <label>
        Bedrooms:
        <input type="number" value={bedrooms()} onInput={(e) => setBedrooms(e.currentTarget.value)} />
      </label>
      {/* <label>
        Age (years):
        <input type="number" value={age()} onInput={(e) => setAge(e.currentTarget.value)} />
      </label> */}
      <button type="submit">Set Input</button>
    </form>
  );
};

export default InputForm;