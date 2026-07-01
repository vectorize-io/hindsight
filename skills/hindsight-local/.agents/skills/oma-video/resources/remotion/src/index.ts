// index.ts — Remotion entry point. registerRoot() is what `npx remotion render
// src/index.ts <CompId> ...` and `remotion studio src/index.ts` look for.
import { registerRoot } from "remotion";
import { RemotionRoot } from "./Root";

registerRoot(RemotionRoot);
