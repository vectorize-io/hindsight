# Frontend Agent - Code Snippets

Copy-paste ready patterns. Use these as starting points, adapt to the specific task.

---

## Next.js 16 framework canonicals (use these, never the legacy alternatives)

```tsx
// Internal nav: <Link>, never <a href="/...">
import Link from "next/link";
<Link href="/gallery" className="...">View gallery</Link>

// Custom font: next/font, never <link rel="stylesheet">
import { Inter } from "next/font/google";
const inter = Inter({ subsets: ["latin"] });
<body className={inter.className}>...</body>

// Images: next/image, never raw <img>
import Image from "next/image";
<Image src="/hero.png" alt="Hero scene" width={1200} height={600} priority />

// Imports: only what you use. After refactoring, remove orphans.
// useCallback / useEffect deps: list every referenced symbol exactly.
```

---

## Accessible Card (focus ring + semantic + keyboard)

Baseline for an interactive surface. Adjust colors via theme tokens; verify
the resulting contrast ratio against the actual `--card` / `--foreground` /
`--muted-foreground` values (theme tokens alone do NOT guarantee 4.5:1; the
designer / token system has to make them so).

```tsx
interface CardProps {
  title: string;
  description?: string;
  onClick?: () => void;
}

export function Card({ title, description, onClick }: CardProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        "rounded-lg border bg-card p-4 text-left shadow-sm transition-colors",
        "hover:bg-accent",
        // visible focus indicator (WCAG 2.4.7 Focus Visible, 2.4.11 Focus Not Obscured)
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
      ].join(" ")}
    >
      {/* Visible text inside the button is the accessible name —
          do NOT add aria-label here, that would override and hide the
          description from screen readers. */}
      <span className="block text-lg font-semibold text-foreground">{title}</span>
      {description && (
        <span className="mt-1 block text-sm text-muted-foreground">
          {description}
        </span>
      )}
    </button>
  );
}
```

Accessibility checklist for interactive surfaces:
- **Semantic element**: `<button>`, `<a>`, `<Link>`. Never `<div onClick>`.
- **Keyboard reachable**: implicit when using semantic elements.
- **Visible focus**: `focus-visible:ring-2 ring-offset-2` (or equivalent). Never strip the outline without replacing it.
- **Accessible name**: visible text inside the element IS the name. Add `aria-label` ONLY for icon-only buttons (e.g., `<button aria-label="Close">×</button>`); when visible text exists, `aria-label` overrides it and is an anti-pattern.
- **Contrast**: verify the actual color values against background reach 4.5:1 (normal text) or 3:1 (large text >= 18pt or 14pt bold). Run an axe/Lighthouse pass; theme tokens are not a proof.
- **Heading semantics**: keep heading tags (`<h1>`-`<h6>`) outside interactive elements. Inside a button, use `<span>` with type-scale classes; promote to a heading at the surrounding section level.

---

## React 19 hook patterns

```tsx
// Derive in render — no state, no effect
function ItemCount({ items }: { items: Item[] }) {
  const count = items.length;
  return <span>{count}</span>;
}

// Initialize once with a lazy initializer
const [id] = useState(() => crypto.randomUUID());

// Pass the ref OBJECT (not its current value) to children that need an instance.
//   The child reads .current inside its own effect or event handler, never in render.
function Selectable({ targetRef }: { targetRef: React.RefObject<THREE.Object3D> }) {
  // ...
}

// Never: useEffect that calls setState to mirror props/state
//    eslint: react-hooks/set-state-in-effect
// useEffect(() => { setCount(items.length); }, [items]);

// Never: gate JSX on ref.current — it is null on first render, and refs
//    do not trigger re-renders when they attach
//    eslint: react-hooks/refs
// return ref.current ? <Child target={ref.current} /> : null;
```

## TanStack Query Hook

```tsx
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

interface Todo {
  id: string;
  title: string;
  completed: boolean;
}

export function useTodos() {
  return useQuery<Todo[]>({
    queryKey: ["todos"],
    queryFn: async () => {
      const res = await fetch("/api/todos", {
        headers: { Authorization: `Bearer ${getToken()}` },
      });
      if (!res.ok) throw new Error("Failed to fetch");
      return res.json();
    },
  });
}

export function useCreateTodo() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data: { title: string }) => {
      const res = await fetch("/api/todos", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${getToken()}`,
        },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error("Failed to create");
      return res.json();
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["todos"] }),
  });
}
```

---

## Form with TanStack Form + Zod

```tsx
"use client";

import { useForm } from "@tanstack/react-form";
import { zodValidator } from "@tanstack/zod-form-adapter";
import { z } from "zod";

const schema = z.object({
  email: z.string().email("Invalid email"),
  password: z.string().min(8, "At least 8 characters"),
});

export function LoginForm({ onSubmit }: { onSubmit: (data: z.infer<typeof schema>) => void }) {
  const form = useForm({
    defaultValues: { email: "", password: "" },
    validatorAdapter: zodValidator(),
    validators: { onChange: schema },
    onSubmit: async ({ value }) => onSubmit(value),
  });

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        e.stopPropagation();
        form.handleSubmit();
      }}
      className="space-y-4"
    >
      <form.Field name="email">
        {(field) => (
          <div>
            <label htmlFor="email" className="text-sm font-medium">
              Email
            </label>
            <input
              id="email"
              type="email"
              value={field.state.value}
              onChange={(e) => field.handleChange(e.target.value)}
              onBlur={field.handleBlur}
              className="mt-1 w-full rounded-md border px-3 py-2"
              aria-invalid={field.state.meta.errors.length > 0}
            />
            {field.state.meta.errors.length > 0 && (
              <p className="mt-1 text-sm text-destructive">{field.state.meta.errors[0]}</p>
            )}
          </div>
        )}
      </form.Field>
      <form.Field name="password">
        {(field) => (
          <div>
            <label htmlFor="password" className="text-sm font-medium">
              Password
            </label>
            <input
              id="password"
              type="password"
              value={field.state.value}
              onChange={(e) => field.handleChange(e.target.value)}
              onBlur={field.handleBlur}
              className="mt-1 w-full rounded-md border px-3 py-2"
              aria-invalid={field.state.meta.errors.length > 0}
            />
            {field.state.meta.errors.length > 0 && (
              <p className="mt-1 text-sm text-destructive">{field.state.meta.errors[0]}</p>
            )}
          </div>
        )}
      </form.Field>
      <form.Subscribe selector={(state) => [state.canSubmit, state.isSubmitting]}>
        {([canSubmit, isSubmitting]) => (
          <button
            type="submit"
            disabled={!canSubmit}
            className="w-full rounded-md bg-primary px-4 py-2 text-primary-foreground disabled:opacity-50"
          >
            {isSubmitting ? "Signing in..." : "Sign in"}
          </button>
        )}
      </form.Subscribe>
    </form>
  );
}
```

---

## Loading / Error / Empty States

```tsx
interface AsyncStateProps<T> {
  data: T | undefined;
  isLoading: boolean;
  error: Error | null;
  empty: React.ReactNode;
  children: (data: T) => React.ReactNode;
}

export function AsyncState<T>({ data, isLoading, error, empty, children }: AsyncStateProps<T>) {
  if (isLoading) return <div className="flex justify-center p-8"><Spinner /></div>;
  if (error) return <ErrorCard message={error.message} />;
  if (!data || (Array.isArray(data) && data.length === 0)) return <>{empty}</>;
  return <>{children(data)}</>;
}
```

---

## Responsive Grid Layout

```tsx
export function StatsGrid({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {children}
    </div>
  );
}
```

---

## Vitest Component Test

```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { Card } from "./Card";

describe("Card", () => {
  it("renders title and description", () => {
    render(<Card title="Test" description="Desc" />);
    expect(screen.getByText("Test")).toBeInTheDocument();
    expect(screen.getByText("Desc")).toBeInTheDocument();
  });

  it("calls onClick when clicked", async () => {
    const onClick = vi.fn();
    render(<Card title="Test" onClick={onClick} />);
    await userEvent.click(screen.getByText("Test"));
    expect(onClick).toHaveBeenCalledOnce();
  });
});
```
