import { createContext, useContext, ParentProps } from "solid-js";
import { Store, AppState } from "./store";

const AppContext = createContext<Store<AppState>>();

export function AppProvider(props: ParentProps<{ store: Store<AppState> }>) {
  return (
    <AppContext.Provider value={props.store}>
      {props.children}
    </AppContext.Provider>
  );
}

export function useAppStore() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error("useAppStore must be used within an AppProvider");
  }
  return context;
}