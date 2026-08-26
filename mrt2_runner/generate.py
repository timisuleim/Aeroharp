"""
generate.py

Reads a control_sequence.json (produced by map_to_mrt2.py), groups
consecutive frames that share the same style_prompt into segments, and
generates one continuous audio clip per segment using Magenta RealTime 2 --
loading the model ONCE so we don't pay the ~55s compile cost per segment.

NOTE: pitch_midi and intensity from the control sequence are not yet used.
MRT2's generate() only accepts a text-style embedding (via embed_style),
not raw MIDI/CC values -- so for now every segment is driven purely by
its style_prompt. Mapping pitch/intensity onto something MRT2 can actually
use (e.g. blending it into the prompt text, or a post-generation gain/pitch
shift) is a separate, still-open design step.

Usage:
    python generate.py control_sequence.json

Output:
    output/<clip_stem>_full.wav          -- all segments concatenated
    output/<clip_stem>_segment_XX.wav    -- each segment individually
"""

import sys
import json
import os
import numpy as np

from magenta_rt import paths
from magenta_rt import MagentaRT2Jax
from magenta_rt.config import MUSICCOCA

CONTROL_FPS = 20  # matches record.py's FPS -- rate control_sequence.json's frames are at
MRT_FPS = 25       # fixed by MagentaRT2Jax.generate()'s frame rate

OUTPUT_DIR = "output"


def load_control_sequence(path):
    with open(path) as f:
        return json.load(f)


def group_into_segments(events):
    """
    Groups consecutive control events that share the same style_prompt.
    Returns a list of dicts: {style_prompt, start_frame, end_frame, num_frames}
    """
    if not events:
        return []

    segments = []
    current_prompt = events[0]["style_prompt"]
    start_frame = events[0]["frame"]
    count = 1

    for event in events[1:]:
        if event["style_prompt"] == current_prompt:
            count += 1
        else:
            segments.append({
                "style_prompt": current_prompt,
                "start_frame": start_frame,
                "num_control_frames": count,
            })
            current_prompt = event["style_prompt"]
            start_frame = event["frame"]
            count = 1

    segments.append({
        "style_prompt": current_prompt,
        "start_frame": start_frame,
        "num_control_frames": count,
    })
    return segments


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate.py path/to/control_sequence.json")
        return

    control_path = sys.argv[1]
    if not os.path.exists(control_path):
        print(f"Control sequence not found: {control_path}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clip_stem = os.path.splitext(os.path.basename(control_path))[0]

    events = load_control_sequence(control_path)
    segments = group_into_segments(events)

    print(f"Loaded {len(events)} control events -> {len(segments)} style segments")
    for i, seg in enumerate(segments):
        duration_sec = seg["num_control_frames"] / CONTROL_FPS
        print(f"  segment {i}: '{seg['style_prompt']}' -- {seg['num_control_frames']} frames "
              f"(~{duration_sec:.2f}s)")

    # Load the model ONCE -- this is the ~55s compile cost, paid a single time
    print("\nLoading MagentaRT2Jax model (this compiles once, ~55s)...")
    mrt = MagentaRT2Jax(
        size=paths.DEFAULT_MODEL_NAME,
        checkpoint=None,
        temperature=1.3,
        top_k=40,
        cfg_scales={"musiccoca": 3.0, "notes": 0.1, "drums": 1.0},
    )
    print("Model loaded.\n")

    segment_wavs = []

    for i, seg in enumerate(segments):
        duration_sec = seg["num_control_frames"] / CONTROL_FPS
        mrt_frames = max(1, int(duration_sec * MRT_FPS))

        print(f"Generating segment {i} ('{seg['style_prompt']}', {mrt_frames} MRT frames)...")
        embedding = mrt.embed_style(seg["style_prompt"], use_mapper=True)
        wav, _state = mrt.generate(
            conditioning={MUSICCOCA.key: embedding},
            frames=mrt_frames,
        )

        segment_path = os.path.join(OUTPUT_DIR, f"{clip_stem}_segment_{i:02d}.wav")
        wav.write(segment_path)
        print(f"  saved {segment_path}")

        segment_wavs.append(wav)

    # Concatenate all segments into one continuous file.
    # NOTE: this is a naive concatenation -- there may be an audible seam at
    # each boundary since we're not passing generation state between segments.
    # Passing the returned `state` from one generate() call into the next
    # would likely produce smoother transitions; worth trying once this
    # baseline version works.
    full_audio = np.concatenate([w.samples for w in segment_wavs], axis=0)
    full_path = os.path.join(OUTPUT_DIR, f"{clip_stem}_full.wav")
    segment_wavs[0].__class__(full_audio, segment_wavs[0].sample_rate).write(full_path)

    print(f"\nDone. Full stitched output: {full_path}")


if __name__ == "__main__":
    main()
