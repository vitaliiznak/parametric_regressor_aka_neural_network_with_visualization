import {
  isServer
} from "./chunk-U3ZE2DDO.js";
import {
  createSignal,
  getListener,
  getOwner,
  onCleanup
} from "./chunk-VMAVG2BC.js";
import "./chunk-BUSYA2B4.js";

// web_app/node_modules/.pnpm/@solid-primitives+scheduled@1.4.3_solid-js@1.8.22/node_modules/@solid-primitives/scheduled/dist/index.js
var debounce = (callback, wait) => {
  if (isServer) {
    return Object.assign(() => void 0, { clear: () => void 0 });
  }
  let timeoutId;
  const clear = () => clearTimeout(timeoutId);
  if (getOwner())
    onCleanup(clear);
  const debounced = (...args) => {
    if (timeoutId !== void 0)
      clear();
    timeoutId = setTimeout(() => callback(...args), wait);
  };
  return Object.assign(debounced, { clear });
};
var throttle = (callback, wait) => {
  if (isServer) {
    return Object.assign(() => void 0, { clear: () => void 0 });
  }
  let isThrottled = false, timeoutId, lastArgs;
  const throttled = (...args) => {
    lastArgs = args;
    if (isThrottled)
      return;
    isThrottled = true;
    timeoutId = setTimeout(() => {
      callback(...lastArgs);
      isThrottled = false;
    }, wait);
  };
  const clear = () => {
    clearTimeout(timeoutId);
    isThrottled = false;
  };
  if (getOwner())
    onCleanup(clear);
  return Object.assign(throttled, { clear });
};
var scheduleIdle = isServer ? () => Object.assign(() => void 0, { clear: () => void 0 }) : (
  // requestIdleCallback is not supported in Safari
  window.requestIdleCallback ? (callback, maxWait) => {
    let isDeferred = false, id, lastArgs;
    const deferred = (...args) => {
      lastArgs = args;
      if (isDeferred)
        return;
      isDeferred = true;
      id = requestIdleCallback(
        () => {
          callback(...lastArgs);
          isDeferred = false;
        },
        { timeout: maxWait }
      );
    };
    const clear = () => {
      cancelIdleCallback(id);
      isDeferred = false;
    };
    if (getOwner())
      onCleanup(clear);
    return Object.assign(deferred, { clear });
  } : (
    // fallback to setTimeout (throttle)
    (callback) => throttle(callback)
  )
);
function leading(schedule, callback, wait) {
  if (isServer) {
    let called = false;
    const scheduled2 = (...args) => {
      if (called)
        return;
      called = true;
      callback(...args);
    };
    return Object.assign(scheduled2, { clear: () => void 0 });
  }
  let isScheduled = false;
  const scheduled = schedule(() => isScheduled = false, wait);
  const func = (...args) => {
    if (!isScheduled)
      callback(...args);
    isScheduled = true;
    scheduled();
  };
  const clear = () => {
    isScheduled = false;
    scheduled.clear();
  };
  if (getOwner())
    onCleanup(clear);
  return Object.assign(func, { clear });
}
function leadingAndTrailing(schedule, callback, wait) {
  if (isServer) {
    let called = false;
    const scheduled2 = (...args) => {
      if (called)
        return;
      called = true;
      callback(...args);
    };
    return Object.assign(scheduled2, { clear: () => void 0 });
  }
  let State;
  ((State2) => {
    State2[State2["Ready"] = 0] = "Ready";
    State2[State2["Leading"] = 1] = "Leading";
    State2[State2["Trailing"] = 2] = "Trailing";
  })(State || (State = {}));
  let state = 0;
  const scheduled = schedule((args) => {
    state === 2 && callback(...args);
    state = 0;
  }, wait);
  const fn = (...args) => {
    if (state !== 2) {
      if (state === 0)
        callback(...args);
      state += 1;
    }
    scheduled(args);
  };
  const clear = () => {
    state = 0;
    scheduled.clear();
  };
  if (getOwner())
    onCleanup(clear);
  return Object.assign(fn, { clear });
}
function createScheduled(schedule) {
  let listeners = 0;
  let isDirty = false;
  const [track, dirty] = createSignal(void 0, { equals: false });
  const call = schedule(() => {
    isDirty = true;
    dirty();
  });
  return () => {
    if (!isDirty)
      call(), track();
    if (isDirty) {
      isDirty = !!listeners;
      return true;
    }
    if (getListener()) {
      listeners++;
      onCleanup(() => listeners--);
    }
    return false;
  };
}
export {
  createScheduled,
  debounce,
  leading,
  leadingAndTrailing,
  scheduleIdle,
  throttle
};
//# sourceMappingURL=@solid-primitives_scheduled.js.map
