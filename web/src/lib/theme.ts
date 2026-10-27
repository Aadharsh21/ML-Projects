import { useSyncExternalStore } from "react";

const query = () => window.matchMedia("(prefers-color-scheme: dark)");

function current(): "light" | "dark" {
  const forced = document.documentElement.dataset.theme;
  if (forced === "light" || forced === "dark") return forced;
  return query().matches ? "dark" : "light";
}

function subscribe(cb: () => void) {
  const mq = query();
  mq.addEventListener("change", cb);
  const observer = new MutationObserver(cb);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  return () => {
    mq.removeEventListener("change", cb);
    observer.disconnect();
  };
}

/** "light" or "dark", following the OS setting unless <html data-theme> forces one. */
export function useColorScheme(): "light" | "dark" {
  return useSyncExternalStore(subscribe, current, () => "light");
}

/** Read a CSS custom property (for canvas drawing). */
export function cssVar(name: string, el: Element = document.documentElement): string {
  return getComputedStyle(el).getPropertyValue(name).trim();
}
