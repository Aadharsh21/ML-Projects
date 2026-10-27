import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const root = (path: string) => fileURLToPath(new URL(path, import.meta.url));
const pages = ["rain", "fraud", "insurance", "iris"];

export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    target: "es2022",
    rollupOptions: {
      input: {
        home: root("./index.html"),
        ...Object.fromEntries(pages.map((p) => [p, root(`./${p}/index.html`)])),
      },
    },
  },
  test: {
    include: ["tests/**/*.test.ts"],
  },
});
