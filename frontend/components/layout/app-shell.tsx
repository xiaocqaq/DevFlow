import Link from "next/link";

const navItems = [
  { href: "/", label: "Workspace" },
  { href: "/knowledge", label: "项目知识" },
  { href: "/rag", label: "RAG 工作室" },
  { href: "/repos", label: "Repos" },
  { href: "/action-drafts", label: "Drafts" },
  { href: "/workflow-runs", label: "Runs" },
  { href: "/workspaces", label: "Reports" },
  { href: "/code-graph", label: "Code Graph" },
  { href: "/evals", label: "Evals" }
];

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <main className="min-h-screen">
      <div className="border-b border-line bg-white/85 px-4 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <div className="mr-2 text-sm font-semibold text-ink">DevFlow AI</div>
          {navItems.map((item) => (
            <Link key={item.href} href={item.href} className="rounded-md px-3 py-1.5 text-sm text-slate-600 transition hover:bg-teal-50 hover:text-teal-800">
              {item.label}
            </Link>
          ))}
        </div>
      </div>
      <div className="w-full p-3">{children}</div>
    </main>
  );
}
