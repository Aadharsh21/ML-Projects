import type { ReactNode } from "react";
import { siteRoot } from "../lib/data";
import { GitHubIcon, NotebookIcon } from "./icons";

export const REPO = "https://github.com/Aadharsh21/ML-Projects";

export type ProjectId = "rain" | "fraud" | "insurance" | "iris";

export const PROJECTS: { id: ProjectId; name: string; title: string }[] = [
  { id: "rain", name: "Rain", title: "Rain tomorrow" },
  { id: "fraud", name: "Fraud", title: "Fraud radar" },
  { id: "insurance", name: "Insurance", title: "Insurance costs" },
  { id: "iris", name: "Clustering", title: "Clustering playground" },
];

export function pageHref(id: ProjectId | "home"): string {
  return id === "home" ? siteRoot() : `${siteRoot()}${id}/`;
}

export function SiteHeader({ current }: { current?: ProjectId }) {
  return (
    <header className="site-header">
      <div className="wrap">
        <a className="wordmark" href={pageHref("home")}>
          <b>Aadharsh Arunn</b>
          <span>Machine learning projects</span>
        </a>
        <nav className="site-nav" aria-label="Projects">
          {PROJECTS.map((p) => (
            <a
              key={p.id}
              href={pageHref(p.id)}
              aria-current={current === p.id ? "page" : undefined}
              style={{ ["--accent" as string]: `var(--${p.id})` }}
            >
              <span className="dot" aria-hidden="true" />
              {p.name}
            </a>
          ))}
          <a href={REPO} aria-label="Source code on GitHub">
            <GitHubIcon />
            Code
          </a>
        </nav>
      </div>
    </header>
  );
}

export function SiteFooter({ children }: { children?: ReactNode }) {
  return (
    <footer className="site-footer">
      <div className="wrap">
        <p>Built by Aadharsh Arunn. Models are trained in Python and run in your browser; nothing is sent to a server.</p>
        {children}
      </div>
    </footer>
  );
}

export function Fact({ value, was, label }: { value: ReactNode; was?: ReactNode; label: ReactNode }) {
  return (
    <div className="fact">
      <div className="value">
        {value}
        {was !== undefined && <span className="was">from {was}</span>}
      </div>
      <div className="label">{label}</div>
    </div>
  );
}

export function PageIntro({
  title,
  lede,
  facts,
  notebook,
  source,
}: {
  title: string;
  lede: ReactNode;
  facts: ReactNode;
  notebook: string;
  source: string;
}) {
  return (
    <section className="intro wrap">
      <h1>{title}</h1>
      <p className="lede">{lede}</p>
      <div className="facts">
        {facts}
        <div className="links">
          <a className="btn" href={`${REPO}/blob/main/notebooks/${notebook}`}>
            <NotebookIcon />
            Notebook
          </a>
          <a className="btn" href={`${REPO}/blob/main/${source}`}>
            <GitHubIcon />
            Model code
          </a>
        </div>
      </div>
    </section>
  );
}

export function Loading({ label, progress }: { label: string; progress: number }) {
  return (
    <div className="loading" role="status" aria-live="polite">
      <div>
        {label}
        <div className="bar">
          <i style={{ width: `${Math.round(progress * 100)}%` }} />
        </div>
      </div>
    </div>
  );
}
