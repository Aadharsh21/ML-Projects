import { existsSync } from "node:fs";
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
      input: Object.fromEntries(
        [["home", root("./index.html")], ...pages.map((p) => [p, root(`./${p}/index.html`)])].filter(([, file]) =>
          existsSync(file),
        ),
      ),
    },
  },
  test: {
    include: ["tests/**/*.test.ts"],
  },
});
