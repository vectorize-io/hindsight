#!/usr/bin/env python3
# pyright: reportMissingImports=false
# The `app.*` imports below are MoneyPrinterTurbo's modules. MPT is never
# vendored into this repo; this driver only ever runs under the MPT checkout's
# own venv (`<MPT>/.venv/bin/python`), where `app` is importable. The repo's
# Python analyzer cannot see them, so this third-party-runtime directive is
# expected — it is not a suppression of any first-party issue.
"""Headless MoneyPrinterTurbo (MPT) driver — boundary-safe subprocess entrypoint.

oma-video never imports MPT. The TypeScript compositor (`internal/mpt-project.ts`
+ `providers/compositor.ts`) spawns *this* script with the MPT venv's python and
a single JSON argument describing the run. The driver loads MPT in-process here
(inside MPT's own venv, never inside oma's runtime) and drives the full pipeline
via `app.services.task.start(...)`, then copies the produced mp4 to the caller's
output path.

Key-free by construction (design 013 §5, backend rule 11):
  * video_script is injected directly -> MPT's custom-script mode, NO LLM key.
  * voice defaults to an edge-tts voice -> NO TTS key.
  * subtitle_provider="edge" (MPT config) reuses the edge sub_maker -> no
    faster-whisper model download.
  * video_source defaults to "local": the driver synthesizes a few ffmpeg
    test-pattern clips into MPT's storage/local_videos so composition needs NO
    Pexels key. video_source="pexels" is used only when a key is provided.

Contract — stdin/argv is one JSON object, stdout's LAST line is one JSON result:
  IN  {
        "mpt_dir":     "<abs path to MPT repo>",      # required
        "script":      "<narration text, one line per scene>",  # required
        "subject":     "<short subject>",             # optional, default "video"
        "out_path":    "<abs output mp4 path>",       # required
        "aspect":      "9:16" | "16:9" | "1:1",       # optional, default 9:16
        "voice_name":  "<edge voice>",                # optional, default en-US
        "video_source":"local" | "pexels",            # optional, default local
        "pexels_api_key": "<key>",                    # optional (pexels source)
        "materials":   ["<abs clip path>", ...],      # optional, local source
        "clip_duration": 5,                            # optional
        "subtitle":    true | false                    # optional, default true
      }
  OUT {"ok": true,  "output": "<abs mp4>", "duration": <float>, "source": "..."}
   |  {"ok": false, "error": "<message>"}

Exit code is 0 on success, 1 on failure; the JSON result line is authoritative.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
import uuid

# Default key-free edge-tts voice. parse_voice_name() strips the gender suffix.
DEFAULT_VOICE = "en-US-AvaNeural-Female"
# MPT's preprocess_video rejects materials below 480x480; portrait shorts need
# height >= width, so 1080x1920 is both safe and matches the design's frame.
RESOLUTIONS = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
}
# A small palette of solid colors so synthesized local clips are visually
# distinct (deterministic order — same input yields the same clips).
COLORS = ["0x1a2332", "0x2d3a4f", "0x3f5168", "0x52687f", "0x6a7f96"]


def _emit(result: dict) -> None:
    """Print the single authoritative JSON result line and flush."""
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()


def _ffmpeg_bin() -> str:
    return os.environ.get("IMAGEIO_FFMPEG_EXE") or shutil.which("ffmpeg") or "ffmpeg"


def _make_test_clip(out_path: str, width: int, height: int, seconds: int,
                    color: str, label: str) -> None:
    """Synthesize a solid-color test clip with ffmpeg (key-free local material).

    Drawn at the target resolution so MPT's >=480x480 check passes and no
    upscaling/letterboxing is required. libx264 + yuv420p keeps the clip a
    standard, broadly decodable mp4.
    """
    ffmpeg = _ffmpeg_bin()
    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi",
        "-i", f"color=c={color}:s={width}x{height}:d={seconds}:r=30",
        "-vf", (
            f"drawtext=text='{label}':fontcolor=white:fontsize={max(width, height)//22}"
            ":x=(w-text_w)/2:y=(h-text_h)/2"
        ),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", str(seconds),
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.isfile(out_path):
        # Fallback without drawtext (some ffmpeg builds lack the freetype filter).
        cmd_plain = [
            ffmpeg, "-y",
            "-f", "lavfi",
            "-i", f"color=c={color}:s={width}x{height}:d={seconds}:r=30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", str(seconds),
            out_path,
        ]
        proc2 = subprocess.run(cmd_plain, capture_output=True, text=True)
        if proc2.returncode != 0 or not os.path.isfile(out_path):
            raise RuntimeError(
                f"ffmpeg failed to synthesize local material: "
                f"{(proc2.stderr or proc.stderr or '').strip()[-400:]}"
            )


def _prepare_local_materials(spec: dict, width: int, height: int,
                             clip_duration: int):
    """Resolve local material clips into MPT's storage/local_videos directory.

    MPT's preprocess_video resolves each material.url *within* storage/local_videos
    (file_security.resolve_path_within_directory). So provided clips are copied in
    and synthesized clips are written there. Returns a list of MaterialInfo.
    """
    from app.models.schema import MaterialInfo  # noqa: E402  (MPT import)
    from app.utils import utils  # noqa: E402

    local_dir = utils.storage_dir("local_videos", create=True)
    materials = []

    # MaterialInfo.url must be an ABSOLUTE path inside storage/local_videos.
    # preprocess_video only rewrites .url for *image* materials; for video clips
    # it returns the original url unchanged, and combine_videos then opens that
    # url directly. A bare filename would be opened relative to cwd and fail
    # ("'oma-synth-00.mp4' not found"). Absolute paths still pass MPT's
    # resolve_path_within_directory guard because they live under local_videos.
    provided = spec.get("materials") or []
    if provided:
        for idx, src in enumerate(provided):
            if not os.path.isfile(src):
                continue
            ext = os.path.splitext(src)[1] or ".mp4"
            dst = os.path.join(local_dir, f"oma-material-{idx:02d}{ext}")
            shutil.copyfile(src, dst)
            materials.append(MaterialInfo(provider="local", url=dst, duration=0))
    if not materials:
        # Synthesize a few deterministic test-pattern clips (key-free path).
        count = 3
        for idx in range(count):
            dst = os.path.join(local_dir, f"oma-synth-{idx:02d}.mp4")
            _make_test_clip(
                dst, width, height, clip_duration,
                COLORS[idx % len(COLORS)], f"scene {idx + 1}",
            )
            materials.append(MaterialInfo(provider="local", url=dst, duration=0))
    return materials


def run(spec: dict) -> dict:
    mpt_dir = spec.get("mpt_dir")
    if not mpt_dir or not os.path.isdir(mpt_dir):
        return {"ok": False, "error": f"mpt_dir not found: {mpt_dir!r}"}
    script = (spec.get("script") or "").strip()
    if not script:
        return {"ok": False, "error": "script is required and must be non-empty"}
    out_path = spec.get("out_path")
    if not out_path:
        return {"ok": False, "error": "out_path is required"}

    aspect = spec.get("aspect") or "9:16"
    if aspect not in RESOLUTIONS:
        aspect = "9:16"
    width, height = RESOLUTIONS[aspect]
    voice_name = spec.get("voice_name") or DEFAULT_VOICE
    video_source = spec.get("video_source") or "local"
    clip_duration = int(spec.get("clip_duration") or 5)
    subtitle = bool(spec.get("subtitle", True))
    subject = spec.get("subject") or "video"

    # Make MPT importable. This driver runs under MPT's OWN venv python, so its
    # third-party deps resolve; we only need MPT's package root on sys.path.
    if mpt_dir not in sys.path:
        sys.path.insert(0, mpt_dir)

    # Pexels key (only when the pexels source is selected) — env-only, never
    # logged. MPT reads pexels keys from its config; set it on config in-memory.
    from app.config import config  # noqa: E402

    if video_source == "pexels":
        key = spec.get("pexels_api_key") or os.environ.get("PEXELS_API_KEY")
        if not key:
            return {"ok": False, "error": "video_source=pexels but no PEXELS_API_KEY"}
        config.app["pexels_api_keys"] = [key]

    from app.models.schema import VideoConcatMode, VideoParams  # noqa: E402
    from app.services import task  # noqa: E402
    from app.utils import utils  # noqa: E402

    video_materials = None
    if video_source == "local":
        video_materials = _prepare_local_materials(
            spec, width, height, clip_duration
        )
        if not video_materials:
            return {"ok": False, "error": "no local materials available"}

    params = VideoParams(
        video_subject=subject,
        # Injecting video_script puts MPT in custom-script mode: generate_script
        # returns it verbatim, no LLM call (backend rule 11 key-free path).
        video_script=script,
        # Empty terms + local source avoids generate_terms' LLM call too.
        video_terms=[] if video_source == "local" else None,
        video_aspect=aspect,
        video_concat_mode=VideoConcatMode.sequential.value,
        video_clip_duration=clip_duration,
        video_count=1,
        video_source=video_source,
        video_materials=video_materials,
        voice_name=voice_name,
        voice_rate=1.0,
        bgm_type="",  # no background music (key-free, deterministic)
        bgm_volume=0.0,
        subtitle_enabled=subtitle,
        n_threads=2,
        paragraph_number=1,
    )

    task_id = "oma-" + uuid.uuid4().hex[:12]
    result = task.start(task_id=task_id, params=params, stop_at="video")
    if not result or not result.get("videos"):
        return {
            "ok": False,
            "error": "MPT task.start produced no videos (see stderr for the MPT log)",
        }

    final = result["videos"][0]
    if not os.path.isfile(final):
        return {"ok": False, "error": f"MPT reported video but file missing: {final}"}

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    shutil.copyfile(final, out_path)

    duration = 0.0
    try:
        duration = float(result.get("audio_duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    # Clean up MPT's per-task storage so the cache clone does not grow unbounded.
    try:
        shutil.rmtree(utils.task_dir(task_id), ignore_errors=True)
    except Exception:  # noqa: BLE001  (cleanup is best-effort)
        pass

    return {
        "ok": True,
        "output": os.path.abspath(out_path),
        "duration": duration,
        "source": video_source,
    }


def main() -> int:
    # Spec JSON comes from argv[1] (a path or inline JSON) or stdin.
    raw = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.isfile(arg):
            with open(arg, "r", encoding="utf-8") as fh:
                raw = fh.read()
        else:
            raw = arg
    if raw is None:
        raw = sys.stdin.read()

    try:
        spec = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        _emit({"ok": False, "error": f"invalid spec JSON: {exc}"})
        return 1

    try:
        result = run(spec)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1

    _emit(result)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
