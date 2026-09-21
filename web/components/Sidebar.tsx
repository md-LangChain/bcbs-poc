"use client";

import type { ThreadSummary } from "@/lib/history";
import { PlusIcon, TrashIcon } from "./Icons";

/** Group chats the way a person thinks about them, newest bucket first. */
function bucket(iso: string): string {
  const t = new Date(iso).getTime();
  if (!iso || Number.isNaN(t)) return "Earlier";
  const days = (Date.now() - t) / 86_400_000;
  if (days < 1) return "Today";
  if (days < 2) return "Yesterday";
  if (days < 7) return "Previous 7 days";
  if (days < 30) return "Previous 30 days";
  return "Earlier";
}

const ORDER = ["Today", "Yesterday", "Previous 7 days", "Previous 30 days", "Earlier"];

export default function Sidebar({
  threads,
  activeId,
  open,
  loading,
  onNew,
  onOpen,
  onDelete,
  onClose,
}: {
  threads: ThreadSummary[];
  activeId: string | null;
  open: boolean;
  loading: boolean;
  onNew: () => void;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
  onClose: () => void;
}) {
  const groups = new Map<string, ThreadSummary[]>();
  for (const t of threads) {
    const key = bucket(t.updatedAt);
    groups.set(key, [...(groups.get(key) ?? []), t]);
  }

  return (
    <>
      <div
        className={`scrim${open ? " show" : ""}`}
        onClick={onClose}
        aria-hidden
      />
      <aside className={`sidebar${open ? " open" : ""}`}>
        <div className="sidebar-top">
          <button className="newchat" onClick={onNew}>
            <PlusIcon />
            New chat
          </button>
        </div>

        <nav className="chatlist" aria-label="Previous chats">
          {threads.length === 0 && (
            <p className="chatlist-empty">
              {loading ? "Loading chats…" : "No chats yet. Review a case to start one."}
            </p>
          )}

          {ORDER.filter((k) => groups.has(k)).map((key) => (
            <section key={key}>
              <div className="chatgroup">{key}</div>
              {groups.get(key)!.map((t) => (
                <div
                  key={t.id}
                  className={`chatrow${t.id === activeId ? " active" : ""}`}
                >
                  <button
                    className="chatrow-open"
                    onClick={() => onOpen(t.id)}
                    title={t.title}
                  >
                    {t.status === "interrupted" && (
                      <span className="pausedot" title="Waiting on your decision" />
                    )}
                    <span className="chatrow-title">{t.title}</span>
                  </button>
                  <button
                    className="chatrow-del"
                    aria-label={`Delete ${t.title}`}
                    title="Delete chat"
                    onClick={() => onDelete(t.id)}
                  >
                    <TrashIcon />
                  </button>
                </div>
              ))}
            </section>
          ))}
        </nav>

        <div className="sidebar-foot">Synthetic data. POC only.</div>
      </aside>
    </>
  );
}
