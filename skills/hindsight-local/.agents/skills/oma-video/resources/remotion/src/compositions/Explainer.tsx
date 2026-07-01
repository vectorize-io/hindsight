// Explainer — 16:9 (or 9:16) explainer composition built from README/code/data.
// Slide + diagram + code frames arrive as render-spec scenes (visual.type
// "slide" | "image"); VideoBase renders them on the timeline.
import { VideoBase } from "../components/VideoBase";
import type { RenderSpec } from "../render-spec";

export const Explainer: React.FC<RenderSpec> = (props) => {
  return <VideoBase {...props} />;
};
