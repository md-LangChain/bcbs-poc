"use client";

import { useEffect, useState } from "react";
import { PanelIcon, PlusIcon, SunMoonIcon } from "./Icons";

export default function Header({
  onNew,
  onToggleSidebar,
}: {
  onNew: () => void;
  onToggleSidebar: () => void;
}) {
  // Rendered only after mount so the button label matches the theme the
  // bootstrap script already applied, without a hydration mismatch.
  const [theme, setTheme] = useState<string | null>(null);

  useEffect(() => {
    setTheme(document.documentElement.dataset.theme ?? "light");
  }, []);

  function toggle() {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    setTheme(next);
    try {
      localStorage.setItem("theme", next);
    } catch {
      // Private browsing / blocked storage — the toggle still works this session.
    }
  }

  return (
    <header className="header">
      <button
        className="iconbtn only-narrow"
        onClick={onToggleSidebar}
        title="Show chats"
        aria-label="Show chats"
      >
        <PanelIcon />
      </button>
      <button
        className="iconbtn only-narrow"
        onClick={onNew}
        title="New chat"
        aria-label="New chat"
      >
        <PlusIcon />
      </button>

      <div className="brand">
        <div
          className="brand-bcbs"
          role="img"
          aria-label="Blue Cross Blue Shield of Rhode Island"
        />
        <span className="brand-x" aria-hidden>
          ×
        </span>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img className="brand-lc" src="/assets/langchain.png" alt="LangChain" />
      </div>
      <div className="header-title">Prior Authorization Review</div>
      <div className="spacer" />
      <button
        className="iconbtn"
        onClick={toggle}
        title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
        aria-label="Toggle colour theme"
      >
        <SunMoonIcon />
      </button>
    </header>
  );
}
