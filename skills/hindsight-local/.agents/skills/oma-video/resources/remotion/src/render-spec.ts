// render-spec.ts — the Remotion-side mirror of the CLI's RenderSpec schema
// (cli/commands/video/types.ts, schemaVersion "1.0"). render-spec.json is the
// deterministic compute boundary: the CLI writes it, `npx remotion render`
// reads it as `--props`. Keep this in sync with the CLI schema; the CLI remains
// the source of truth.
import { z } from "zod";

export const VIDEO_SCHEMA_VERSION = "1.0" as const;

export const CaptionStyleSchema = z.enum(["tiktok", "lower-third", "none"]);

export const SafeAreaSchema = z.object({
  topPct: z.number().nonnegative(),
  bottomPct: z.number().nonnegative(),
  leftPct: z.number().nonnegative(),
  rightPct: z.number().nonnegative(),
});

export const RenderSpecSceneSchema = z.object({
  id: z.string().min(1),
  fromFrame: z.number().int().nonnegative(),
  durationInFrames: z.number().int().positive(),
  visual: z.object({
    type: z.enum(["image", "video", "slide", "capture", "placeholder"]),
    src: z.string().min(1),
    kenBurns: z.boolean().default(false),
  }),
  onScreenText: z.array(z.string()).default([]),
  transitionOut: z.string().optional(),
});

export const RenderSpecSchema = z.object({
  schemaVersion: z.literal(VIDEO_SCHEMA_VERSION),
  compositor: z.enum(["remotion", "mpt"]),
  composition: z.string().min(1),
  fps: z.number().int().positive(),
  dimensions: z.object({
    width: z.number().int().positive(),
    height: z.number().int().positive(),
  }),
  durationInFrames: z.number().int().nonnegative(),
  audio: z.object({
    narration: z.string().optional(),
    music: z.string().optional(),
    musicGainDb: z.number().optional(),
  }),
  scenes: z.array(RenderSpecSceneSchema),
  captions: z.object({
    file: z.string().optional(),
    style: CaptionStyleSchema,
    fontFamily: z.string(),
    maxWidthPct: z.number().positive().max(100),
    safeArea: SafeAreaSchema,
  }),
  background: z.object({
    type: z.enum(["color", "image", "video"]),
    src: z.string().optional(),
  }),
  seed: z.number().int(),
});

export type RenderSpec = z.infer<typeof RenderSpecSchema>;
export type RenderSpecScene = z.infer<typeof RenderSpecSceneSchema>;
export type SafeArea = z.infer<typeof SafeAreaSchema>;

// Default props used by the Remotion Studio preview + as the schema fallback.
// A real render overrides every field via --props=render-spec.json.
export const PLACEHOLDER_RENDER_SPEC: RenderSpec = {
  schemaVersion: VIDEO_SCHEMA_VERSION,
  compositor: "remotion",
  composition: "Shorts",
  fps: 30,
  dimensions: { width: 1080, height: 1920 },
  durationInFrames: 90,
  audio: {},
  scenes: [
    {
      id: "scene-01",
      fromFrame: 0,
      durationInFrames: 90,
      visual: { type: "placeholder", src: "#0f1117", kenBurns: false },
      onScreenText: ["oma-video"],
    },
  ],
  captions: {
    style: "tiktok",
    fontFamily: "Pretendard",
    maxWidthPct: 86,
    safeArea: { topPct: 8, bottomPct: 18, leftPct: 7, rightPct: 7 },
  },
  background: { type: "color", src: "#0f1117" },
  seed: 1,
};
