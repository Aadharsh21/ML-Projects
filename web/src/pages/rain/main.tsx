import "@fontsource-variable/archivo/wdth.css";
import "@fontsource-variable/newsreader/opsz.css";
import "../../styles/tokens.css";
import "../../styles/base.css";
import "./rain.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { RainApp } from "./RainApp";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RainApp />
  </StrictMode>,
);
