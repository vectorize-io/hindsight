// Demo — 16:9 demo/walkthrough composition over a human-recorded capture.
// The capture arrives as a render-spec scene/background (visual.type "capture");
// intro card + callout scenes are additional render-spec scenes. Zoom/callout
// motion is expressed as Ken Burns + on-screen text on those scenes, so the
// timeline stays a pure function of the render-spec.
import { VideoBase } from "../components/VideoBase";
import type { RenderSpec } from "../render-spec";

export const Demo: React.FC<RenderSpec> = (props) => {
  return <VideoBase {...props} />;
};
