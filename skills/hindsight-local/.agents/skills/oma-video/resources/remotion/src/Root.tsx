// Root.tsx — registers the Shorts / Explainer / Demo compositions.
//
// Each <Composition> reads render-spec.json (passed via `--props`) as input
// props. dimensions/fps/durationInFrames come FROM the render-spec, so we use
// calculateMetadata to override the static defaults at render time. The Zod
// `schema` validates the props, giving the CLI adapter a typed contract: an
// invalid render-spec fails fast (maps to the CLI's SchemaValidationError / exit
// 4) instead of rendering garbage.
import { Composition, type CalculateMetadataFunction } from "remotion";
import { Shorts } from "./compositions/Shorts";
import { Explainer } from "./compositions/Explainer";
import { Demo } from "./compositions/Demo";
import {
  RenderSpecSchema,
  PLACEHOLDER_RENDER_SPEC,
  type RenderSpec,
} from "./render-spec";

// Derive real dimensions/fps/duration from the render-spec props.
const calculateMetadata: CalculateMetadataFunction<RenderSpec> = ({ props }) => {
  return {
    width: props.dimensions.width,
    height: props.dimensions.height,
    fps: props.fps,
    durationInFrames: Math.max(1, props.durationInFrames),
  };
};

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="Shorts"
        component={Shorts}
        schema={RenderSpecSchema}
        defaultProps={{ ...PLACEHOLDER_RENDER_SPEC, composition: "Shorts" }}
        calculateMetadata={calculateMetadata}
        // Static fallbacks (9:16); overridden by calculateMetadata at render.
        width={1080}
        height={1920}
        fps={30}
        durationInFrames={90}
      />
      <Composition
        id="Explainer"
        component={Explainer}
        schema={RenderSpecSchema}
        defaultProps={{
          ...PLACEHOLDER_RENDER_SPEC,
          composition: "Explainer",
          dimensions: { width: 1920, height: 1080 },
        }}
        calculateMetadata={calculateMetadata}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={90}
      />
      <Composition
        id="Demo"
        component={Demo}
        schema={RenderSpecSchema}
        defaultProps={{
          ...PLACEHOLDER_RENDER_SPEC,
          composition: "Demo",
          dimensions: { width: 1920, height: 1080 },
        }}
        calculateMetadata={calculateMetadata}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={90}
      />
    </>
  );
};
