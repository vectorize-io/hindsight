// Scene.tsx — render one render-spec scene: a visual (image / video / slide /
// capture / placeholder) plus its on-screen text, with optional Ken Burns.
//
// Each scene is placed on the timeline by the composition using <Sequence>; this
// component only draws the visual for the duration it is mounted. Ken Burns is a
// deterministic slow zoom driven purely by the frame (no randomness), so the
// output is reproducible from the render-spec + seed.
import {
  AbsoluteFill,
  Img,
  OffthreadVideo,
  staticFile,
  interpolate,
  useCurrentFrame,
} from "remotion";
import { FONT_STACK } from "../load-fonts";
import type { RenderSpecScene } from "../render-spec";

const isColor = (src: string): boolean => src.startsWith("#");

export const Scene: React.FC<{ scene: RenderSpecScene }> = ({ scene }) => {
  const frame = useCurrentFrame();
  const { type, src, kenBurns } = scene.visual;

  // Deterministic slow zoom over the scene's local frame range.
  const scale = kenBurns
    ? interpolate(frame, [0, scene.durationInFrames], [1, 1.08], {
        extrapolateRight: "clamp",
      })
    : 1;

  return (
    <AbsoluteFill>
      <AbsoluteFill style={{ transform: `scale(${scale})` }}>
        {renderVisual(type, src)}
      </AbsoluteFill>
      {scene.onScreenText.length > 0 ? (
        <AbsoluteFill
          style={{
            justifyContent: "flex-start",
            alignItems: "center",
            paddingTop: "10%",
          }}
        >
          <div
            style={{
              fontFamily: FONT_STACK,
              fontWeight: 800,
              fontSize: 56,
              color: "#ffffff",
              textShadow: "0 2px 10px rgba(0,0,0,0.8)",
              textAlign: "center",
              maxWidth: "86%",
            }}
          >
            {scene.onScreenText.join("\n")}
          </div>
        </AbsoluteFill>
      ) : null}
    </AbsoluteFill>
  );
};

function renderVisual(
  type: RenderSpecScene["visual"]["type"],
  src: string,
): React.ReactNode {
  if (type === "placeholder" || (type === "image" && isColor(src))) {
    const color = isColor(src) ? src : "#0f1117";
    return <AbsoluteFill style={{ backgroundColor: color }} />;
  }
  if (type === "video" || type === "capture") {
    return (
      <OffthreadVideo
        src={staticFile(src)}
        style={{ width: "100%", height: "100%", objectFit: "cover" }}
      />
    );
  }
  // image | slide -> still frame
  return (
    <Img
      src={staticFile(src)}
      style={{ width: "100%", height: "100%", objectFit: "cover" }}
    />
  );
}
