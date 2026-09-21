import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Prior Authorization Review",
  description:
    "Blue Cross Blue Shield of Rhode Island × LangChain — agentic prior authorization review",
};

/**
 * Applied before first paint so the correct theme is on <html> immediately and
 * the page never flashes the wrong one.
 */
const THEME_BOOTSTRAP = `
(function () {
  try {
    var forced = new URLSearchParams(location.search).get("theme");
    var saved = forced || localStorage.getItem("theme");
    var dark = saved ? saved === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "dark" : "light";
  } catch (e) {
    document.documentElement.dataset.theme = "light";
  }
})();
`;

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOTSTRAP }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
