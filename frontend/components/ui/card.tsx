import { clsx } from "clsx";

export function Card({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <section className={clsx("rounded-lg border border-line/80 bg-white/90 p-5 shadow-sm shadow-slate-200/60 backdrop-blur", className)}>
      {children}
    </section>
  );
}

export function PageTitle({ title, description }: { title: string; description: string }) {
  return (
    <div className="mb-6 flex flex-col gap-2 border-b border-line/80 pb-5">
      <div className="text-xs font-medium uppercase tracking-[0.18em] text-accent">DevFlow AI</div>
      <h1 className="text-2xl font-semibold tracking-normal text-ink md:text-3xl">{title}</h1>
      <p className="max-w-3xl text-sm leading-6 text-slate-600">{description}</p>
    </div>
  );
}

export function PrimaryButton({ children, className, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={clsx(
        "inline-flex items-center justify-center rounded-md bg-accent px-4 py-2 text-sm font-medium text-white shadow-sm shadow-teal-900/10 transition hover:bg-teal-700 disabled:cursor-not-allowed disabled:opacity-60",
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}

export function GhostButton({ children, className, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={clsx(
        "w-full rounded-md border border-line bg-white px-3 py-2 text-left text-sm text-slate-700 transition hover:border-teal-200 hover:bg-teal-50/70 hover:text-teal-900",
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}
