"use client";

export default function GlobalError({
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="en">
      <body
        style={{
          background: "#0b0f17",
          color: "#e6edf6",
          fontFamily: "system-ui, sans-serif",
          padding: "3rem",
        }}
      >
        <h2>Something went wrong</h2>
        <p style={{ color: "#8b9bb4" }}>The Mission Sandbox UI hit an unexpected error.</p>
        <button
          onClick={() => reset()}
          style={{
            marginTop: "1rem",
            borderRadius: "0.5rem",
            background: "#4c8dff",
            color: "white",
            padding: "0.5rem 1rem",
            border: 0,
          }}
        >
          Try again
        </button>
      </body>
    </html>
  );
}
