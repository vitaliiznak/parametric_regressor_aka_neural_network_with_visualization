import { createContext, useContext, ParentProps } from "solid-js";
import { Store, SetStoreFunction } from "solid-js/store";
import { AppState } from "./store";

const AppContext = createContext<[Store<AppState>, SetStoreFunction<AppState>]>();

export function AppProvider(props: ParentProps<{ store: [Store<AppState>, SetStoreFunction<AppState>] }>) {
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