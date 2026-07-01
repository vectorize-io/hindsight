// Captions.tsx — render captions.srt over the video using @remotion/captions.
//
// The CLI's oma-captions provider writes captions.srt (+ .vtt) into the run
// dir; render-spec.captions.file points at it. Here we fetch + parseSrt() the
// .srt and display the cue active at the current frame. Two styles map from
// render-spec.captions.style:
//   - "tiktok"      : centered, large, animated pop, sits in the lower-third
//                     above the safe-area bottom margin.
//   - "lower-third" : smaller, left-aligned band near the bottom.
//   - "none"        : nothing rendered (handled by the caller).
//
// We window on the raw SRT cues (one per scene narration), NOT on
// createTikTokStyleCaptions pages: our cues are sentence-level and back-to-back
// (zero gap), so token-combining merged every cue into a single page and dumped
// the entire script on screen at once. Showing the active cue is correct for
// sentence-level captions and degrades gracefully for any style.
import { useCallback, useEffect, useMemo, useState } from "react";
import { parseSrt, type Caption } from "@remotion/captions";
import {
  AbsoluteFill,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
  delayRender,
  continueRender,
  cancelRender,
} from "remotion";
import { FONT_STACK } from "../load-fonts";
import type { CaptionStyleSchema, SafeArea } from "../render-spec";
import type { z } from "zod";

type CaptionStyle = z.infer<typeof CaptionStyleSchema>;

export const Captions: React.FC<{
  file?: string;
  style: CaptionStyle;
  maxWidthPct: number;
  safeArea: SafeArea;
}> = ({ file, style, maxWidthPct, safeArea }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const [cues, setCues] = useState<Caption[] | null>(null);
  const [handle] = useState(() => delayRender("Loading captions"));

  const fetchCaptions = useCallback(async () => {
    if (!file) {
      setCues([]);
      continueRender(handle);
      return;
    }
    try {
      const res = await fetch(staticFile(file));
      const text = await res.text();
      const { captions } = parseSrt({ input: text }) as { captions: Caption[] };
      setCues(captions);
      continueRender(handle);
    } catch (err) {
      cancelRender(err);
    }
  }, [file, handle]);

  useEffect(() => {
    fetchCaptions();
  }, [fetchCaptions]);

  const activeCue = useMemo<Caption | null>(() => {
    if (!cues) return null;
    const nowMs = (frame / fps) * 1000;
    return (
      cues.find((cue) => nowMs >= cue.startMs && nowMs < cue.endMs) ?? null
    );
  }, [cues, frame, fps]);

  if (style === "none" || !activeCue) return null;

  const isTikTok = style === "tiktok";
  return (
    <AbsoluteFill
      style={{
        justifyContent: "flex-end",
        alignItems: isTikTok ? "center" : "flex-start",
        paddingBottom: `${safeArea.bottomPct}%`,
        paddingLeft: `${safeArea.leftPct}%`,
        paddingRight: `${safeArea.rightPct}%`,
        pointerEvents: "none",
      }}
    >
      <div
        style={{
          maxWidth: `${maxWidthPct}%`,
          fontFamily: FONT_STACK,
          fontWeight: 800,
          textAlign: isTikTok ? "center" : "left",
          color: "#ffffff",
          textShadow: "0 2px 8px rgba(0,0,0,0.85)",
          fontSize: isTikTok ? 64 : 40,
          lineHeight: 1.15,
          letterSpacing: -0.5,
          background: isTikTok ? "transparent" : "rgba(0,0,0,0.55)",
          padding: isTikTok ? 0 : "12px 20px",
          borderRadius: isTikTok ? 0 : 12,
        }}
      >
        {activeCue.text}
      </div>
    </AbsoluteFill>
  );
};
