// Remotion CLI config for the vendored oma-video compositor.
// Kept minimal + deterministic: H.264 mp4, overwrite output, color-managed.
// The CLI adapter passes --props=render-spec.json; this file only fixes the
// encoder/output defaults so renders are reproducible across machines.
import { Config } from "@remotion/cli/config";

Config.setVideoImageFormat("jpeg");
Config.setCodec("h264");
Config.setOverwriteOutput(true);
Config.setChromiumOpenGlRenderer("angle");
