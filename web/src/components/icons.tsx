import type { SVGProps } from "react";

type P = SVGProps<SVGSVGElement>;
const base = { width: 16, height: 16, viewBox: "0 0 16 16", fill: "none", "aria-hidden": true } as const;

export const PlayIcon = (p: P) => (
  <svg {...base} {...p}><path d="M4.5 2.8v10.4L13 8 4.5 2.8Z" fill="currentColor" /></svg>
);
export const PauseIcon = (p: P) => (
  <svg {...base} {...p}><path d="M4.5 3h2.5v10H4.5zM9 3h2.5v10H9z" fill="currentColor" /></svg>
);
export const StepIcon = (p: P) => (
  <svg {...base} {...p}><path d="M3.5 3v10L10 8 3.5 3Z" fill="currentColor" /><path d="M11.5 3h1.6v10h-1.6z" fill="currentColor" /></svg>
);
export const ResetIcon = (p: P) => (
  <svg {...base} {...p}><path d="M3.2 8a4.8 4.8 0 1 0 1.5-3.5M3.2 2.6v2.9h2.9" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
);
export const LocateIcon = (p: P) => (
  <svg {...base} {...p}><circle cx="8" cy="8" r="4" stroke="currentColor" strokeWidth="1.6" /><path d="M8 1v2.5M8 12.5V15M1 8h2.5M12.5 8H15" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg>
);
export const DropIcon = (p: P) => (
  <svg {...base} {...p}><path d="M8 1.8c2.4 3 4.2 5.4 4.2 7.4a4.2 4.2 0 1 1-8.4 0c0-2 1.8-4.4 4.2-7.4Z" fill="currentColor" /></svg>
);
export const SunIcon = (p: P) => (
  <svg {...base} {...p}><circle cx="8" cy="8" r="3" fill="currentColor" /><path d="M8 1v2M8 13v2M1 8h2M13 8h2M3 3l1.4 1.4M11.6 11.6 13 13M3 13l1.4-1.4M11.6 4.4 13 3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" /></svg>
);
export const CheckIcon = (p: P) => (
  <svg {...base} {...p}><path d="m3 8.5 3.2 3L13 4.8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
);
export const CrossIcon = (p: P) => (
  <svg {...base} {...p}><path d="m4 4 8 8M12 4l-8 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>
);
export const AlertIcon = (p: P) => (
  <svg {...base} {...p}><path d="M8 1.8 15 14H1L8 1.8Z" fill="currentColor" /><path d="M8 6v3.6" stroke="var(--page)" strokeWidth="1.6" strokeLinecap="round" /><circle cx="8" cy="11.6" r="0.9" fill="var(--page)" /></svg>
);
export const GitHubIcon = (p: P) => (
  <svg {...base} {...p}><path fill="currentColor" d="M8 .2a8 8 0 0 0-2.5 15.6c.4 0 .5-.2.5-.4v-1.5c-2.2.5-2.7-1-2.7-1-.4-.9-.9-1.2-.9-1.2-.7-.5.1-.5.1-.5.8.1 1.2.8 1.2.8.7 1.3 1.9.9 2.4.7 0-.5.3-.9.5-1.1-1.8-.2-3.6-.9-3.6-3.9 0-.9.3-1.6.8-2.1-.1-.2-.4-1 .1-2.1 0 0 .7-.2 2.2.8a7.6 7.6 0 0 1 4 0c1.5-1 2.2-.8 2.2-.8.4 1.1.2 1.9.1 2.1.5.6.8 1.3.8 2.1 0 3-1.9 3.7-3.6 3.9.3.3.5.8.5 1.5v2.2c0 .2.1.5.6.4A8 8 0 0 0 8 .2Z" /></svg>
);
export const NotebookIcon = (p: P) => (
  <svg {...base} {...p}><rect x="3" y="1.8" width="10" height="12.4" rx="1.5" stroke="currentColor" strokeWidth="1.5" /><path d="M6 5h4.5M6 8h4.5M6 11h2.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" /></svg>
);
export const UploadIcon = (p: P) => (
  <svg {...base} {...p}><path d="M8 11V2.5M4.5 6 8 2.5 11.5 6M2.5 11v2.5h11V11" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
);
export const DownloadIcon = (p: P) => (
  <svg {...base} {...p}><path d="M8 2.5V11M4.5 7.5 8 11l3.5-3.5M2.5 13.5h11" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
);
export const LinkIcon = (p: P) => (
  <svg {...base} {...p}><path d="M6.5 9.5 9.5 6.5M7 4.2l1-1a2.8 2.8 0 0 1 4 4l-1 1M9 11.8l-1 1a2.8 2.8 0 0 1-4-4l1-1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" /></svg>
);
