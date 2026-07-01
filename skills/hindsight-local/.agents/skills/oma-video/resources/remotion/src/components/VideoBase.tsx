// VideoBase.tsx — the shared timeline used by all three modes. It lays out the
// render-spec scenes with <Sequence>, draws the background, mounts narration +
// optional music <Audio>, and overlays captions. Shorts/Explainer/Demo are thin
// wrappers that only differ in default framing; the timeline logic is one place.
import {
  AbsoluteFill,
  Audio,
  Sequence,
  staticFile,
} from "remotion";
import { Scene } from "./Scene";
import { Captions } from "./Captions";
import { ensurePretendard } from "../load-fonts";
import type { RenderSpec } from "../render-spec";

// Block the render until Pretendard is ready (deterministic glyphs).
void ensurePretendard();

export const VideoBase: React.FC<RenderSpec> = (spec) => {
  const bgColor =
    spec.background.type === "color" ? spec.background.src ?? "#0f1117" : "#000";

  return (
    <AbsoluteFill style={{ backgroundColor: bgColor }}>
      {spec.background.type !== "color" && spec.background.src ? (
        <AbsoluteFill>
          <Scene
            scene={{
              id: "bg",
              fromFrame: 0,
              durationInFrames: spec.durationInFrames,
              visual: {
                type: spec.background.type === "video" ? "video" : "image",
                src: spec.background.src,
                kenBurns: false,
              },
              onScreenText: [],
            }}
          />
        </AbsoluteFill>
      ) : null}

      {spec.scenes.map((scene) => (
        <Sequence
          key={scene.id}
          from={scene.fromFrame}
          durationInFrames={scene.durationInFrames}
          name={scene.id}
        >
          <Scene scene={scene} />
        </Sequence>
      ))}

      {spec.audio.narration ? (
        <Audio src={staticFile(spec.audio.narration)} />
      ) : null}
      {spec.audio.music ? (
        <Audio
          src={staticFile(spec.audio.music)}
          volume={dbToGain(spec.audio.musicGainDb ?? -18)}
        />
      ) : null}

      <Captions
        file={spec.captions.file}
        style={spec.captions.style}
        maxWidthPct={spec.captions.maxWidthPct}
        safeArea={spec.captions.safeArea}
      />
    </AbsoluteFill>
  );
};

function dbToGain(db: number): number {
  return Math.min(1, Math.max(0, 10 ** (db / 20)));
}
