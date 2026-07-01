// Shorts — 9:16 short-form composition. Thin wrapper over VideoBase; all
// timeline/visual/caption logic lives there. The render-spec drives dimensions,
// fps, and duration via the <Composition> calculateMetadata in Root.tsx.
import { VideoBase } from "../components/VideoBase";
import type { RenderSpec } from "../render-spec";

export const Shorts: React.FC<RenderSpec> = (props) => {
  return <VideoBase {...props} />;
};
